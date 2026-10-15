(declaim (optimize (speed 3) (safety 0) (space 0) (debug 0)))

;; Detect OS and load appropriate shared library
(defvar *lib-path*
  (let* ((os (software-type))
         (lib-name (if (string-equal os "Darwin") "libc_loader.dylib" "libc_loader.so"))
         (paths (list lib-name
                      (format nil "src/~A" lib-name)
                      (format nil "lisp_project/src/~A" lib-name)
                      (format nil "mojo_project/src/~A" lib-name))))
    (or (find-if #'probe-file paths)
        lib-name)))

(handler-case
    (sb-alien:load-shared-object *lib-path* :dont-save t)
  (error (e)
    (format *error-output* "Warning loading shared library: ~A~%" e)))

(sb-alien:define-alien-routine ("c_get_image_dims" c-get-image-dims) sb-alien:int
  (path sb-alien:c-string)
  (w (* sb-alien:int))
  (h (* sb-alien:int))
  (ch (* sb-alien:int)))

(sb-alien:define-alien-routine ("c_load_image_to_ptr" c-load-image-to-ptr) sb-alien:int
  (path sb-alien:c-string)
  (dst (* sb-alien:unsigned-char)))

(sb-alien:define-alien-routine ("c_save_image_from_ptr" c-save-image-from-ptr) sb-alien:void
  (path sb-alien:c-string)
  (w sb-alien:int)
  (h sb-alien:int)
  (ch sb-alien:int)
  (src (* sb-alien:unsigned-char)))

(sb-alien:define-alien-routine ("c_get_time" c-get-time) sb-alien:double-float)
(sb-alien:define-alien-routine ("c_get_num_procs" c-get-num-procs) sb-alien:int)

;; Separable Horizontal Blur on slice [y-start, y-end)
(defun horizontal-blur-range (in-ptr tmp-ptr width height channels y-start y-end)
  (declare (ignore height))
  (declare (type (sb-alien:alien (* sb-alien:unsigned-char)) in-ptr tmp-ptr)
           (type fixnum width channels y-start y-end))
  (let ((row-len (* width channels)))
    (declare (type fixnum row-len))
    (loop for y of-type fixnum from y-start below y-end do
      (let ((y-off (* y row-len)))
        (declare (type fixnum y-off))
        (loop for x of-type fixnum from 0 below width do
          (let ((x0 (if (> x 0) (the fixnum (1- x)) 0))
                (x1 x)
                (x2 (if (< (the fixnum (1+ x)) width) (the fixnum (1+ x)) (the fixnum (1- width)))))
            (declare (type fixnum x0 x1 x2))
            (loop for c of-type fixnum from 0 below channels do
              (let* ((p0 (sb-alien:deref in-ptr (+ y-off (* x0 channels) c)))
                     (p1 (sb-alien:deref in-ptr (+ y-off (* x1 channels) c)))
                     (p2 (sb-alien:deref in-ptr (+ y-off (* x2 channels) c)))
                     (sum (+ p0 (ash p1 1) p2 2))
                     (r (ash sum -2)))
                (declare (type fixnum p0 p1 p2 sum r))
                (setf (sb-alien:deref tmp-ptr (+ y-off (* x channels) c)) r)))))))))

;; Separable Vertical Blur on slice [y-start, y-end)
(defun vertical-blur-range (tmp-ptr out-ptr width height channels y-start y-end)
  (declare (type (sb-alien:alien (* sb-alien:unsigned-char)) tmp-ptr out-ptr)
           (type fixnum width height channels y-start y-end))
  (let ((row-len (* width channels)))
    (declare (type fixnum row-len))
    (loop for y of-type fixnum from y-start below y-end do
      (let* ((y0 (if (> y 0) (the fixnum (1- y)) 0))
             (y1 y)
             (y2 (if (< (the fixnum (1+ y)) height) (the fixnum (1+ y)) (the fixnum (1- height))))
             (y0-off (* y0 row-len))
             (y1-off (* y1 row-len))
             (y2-off (* y2 row-len))
             (out-y-off (* y row-len)))
        (declare (type fixnum y0 y1 y2 y0-off y1-off y2-off out-y-off))
        (loop for x of-type fixnum from 0 below width do
          (let ((x-off (* x channels)))
            (declare (type fixnum x-off))
            (loop for c of-type fixnum from 0 below channels do
              (let* ((r0 (sb-alien:deref tmp-ptr (+ y0-off x-off c)))
                     (r1 (sb-alien:deref tmp-ptr (+ y1-off x-off c)))
                     (r2 (sb-alien:deref tmp-ptr (+ y2-off x-off c)))
                     (sum (+ r0 (ash r1 1) r2 2))
                     (res (ash sum -2)))
                (declare (type fixnum r0 r1 r2 sum res))
                (setf (sb-alien:deref out-ptr (+ out-y-off x-off c)) res)))))))))

(defun run-convolution (technique in-path out-path)
  (sb-alien:with-alien ((w sb-alien:int 0)
                       (h sb-alien:int 0)
                       (ch sb-alien:int 0))
    (let ((ok (c-get-image-dims in-path (sb-alien:addr w) (sb-alien:addr h) (sb-alien:addr ch))))
      (when (zerop ok)
        (format *error-output* "Failed to open image ~A~%" in-path)
        (sb-ext:exit :code 1))

      (let* ((width w)
             (height h)
             (channels ch)
             (total-bytes (* width height channels))
             (in-ptr (sb-alien:make-alien sb-alien:unsigned-char total-bytes))
             (tmp-ptr (sb-alien:make-alien sb-alien:unsigned-char total-bytes))
             (out-ptr (sb-alien:make-alien sb-alien:unsigned-char total-bytes)))

        (c-load-image-to-ptr in-path in-ptr)

        (let* ((env-threads (sb-ext:posix-getenv "OMP_NUM_THREADS"))
               (max-threads (if (and env-threads (> (length env-threads) 0))
                                (or (parse-integer env-threads :junk-allowed t) (c-get-num-procs))
                                (c-get-num-procs)))
               (num-threads (if (or (= technique 3) (= technique 1)) (max 1 max-threads) 1))
               (chunk-size (ceiling height num-threads))
               (t0 (c-get-time)))

          (if (> num-threads 1)
              (let ((threads nil))
                ;; Multi-threaded horizontal pass
                (dotimes (i num-threads)
                  (let ((ys (* i chunk-size))
                        (ye (min (* (1+ i) chunk-size) height)))
                    (when (< ys height)
                      (push (sb-thread:make-thread
                             (lambda ()
                               (horizontal-blur-range in-ptr tmp-ptr width height channels ys ye))
                             :name (format nil "hblur-~A" i))
                            threads))))
                (mapc #'sb-thread:join-thread threads)

                ;; Multi-threaded vertical pass
                (setf threads nil)
                (dotimes (i num-threads)
                  (let ((ys (* i chunk-size))
                        (ye (min (* (1+ i) chunk-size) height)))
                    (when (< ys height)
                      (push (sb-thread:make-thread
                             (lambda ()
                               (vertical-blur-range tmp-ptr out-ptr width height channels ys ye))
                             :name (format nil "vblur-~A" i))
                            threads))))
                (mapc #'sb-thread:join-thread threads))

              ;; Single-threaded baseline
              (progn
                (horizontal-blur-range in-ptr tmp-ptr width height channels 0 height)
                (vertical-blur-range tmp-ptr out-ptr width height channels 0 height)))

          (let ((elapsed (- (c-get-time) t0)))
            (when (string/= out-path "none")
              (c-save-image-from-ptr out-path width height channels out-ptr))

            (sb-alien:free-alien in-ptr)
            (sb-alien:free-alien tmp-ptr)
            (sb-alien:free-alien out-ptr)

            (format t "EXECUTION_TIME: ~,6F~%" elapsed)
            (finish-output)))))))

(defun ensure-lib-loaded ()
  (let* ((os (software-type))
         (lib-name (if (string-equal os "Darwin") "libc_loader.dylib" "libc_loader.so"))
         (exe (or (first sb-ext:*posix-argv*) ""))
         (exe-dir (if (probe-file exe) (directory-namestring (truename exe)) ""))
         (candidates (list (format nil "~A../src/~A" exe-dir lib-name)
                           (format nil "~Asrc/~A" exe-dir lib-name)
                           (format nil "~A~A" exe-dir lib-name)
                           (format nil "lisp_project/src/~A" lib-name)
                           (format nil "src/~A" lib-name)
                           lib-name))
         (found (find-if #'probe-file candidates)))
    (if found
        (handler-case
            (sb-alien:load-shared-object (namestring (truename found)))
          (error (e)
            (format *error-output* "Error loading ~A: ~A~%" found e)))
        (format *error-output* "Warning: could not find ~A~%" lib-name))))

(defun main ()
  (ensure-lib-loaded)
  (let* ((args (cdr sb-ext:*posix-argv*))
         (argc (length args)))
    (if (< argc 3)
        (progn
          (format *error-output* "Usage: lisp_conv <technique 1-4> <input_image> <output_image>~%")
          (sb-ext:exit :code 1))
        (let ((tech (parse-integer (first args)))
              (in-path (second args))
              (out-path (third args)))
          (run-convolution tech in-path out-path)
          (sb-ext:exit :code 0)))))
