program convolution_fortran
    use iso_c_binding
    implicit none

    interface
        function load_image_c(filename, width, height, channels) bind(c, name="load_image_c")
            import :: c_ptr, c_char, c_int
            character(kind=c_char), intent(in) :: filename(*)
            integer(c_int), intent(out) :: width, height, channels
            type(c_ptr) :: load_image_c
        end function load_image_c

        subroutine free_image_c(pdata) bind(c, name="free_image_c")
            import :: c_ptr
            type(c_ptr), value :: pdata
        end subroutine free_image_c

        subroutine save_image_c(filename, width, height, channels, pdata) bind(c, name="save_image_c")
            import :: c_ptr, c_char, c_int
            character(kind=c_char), intent(in) :: filename(*)
            integer(c_int), value :: width, height, channels
            type(c_ptr), value :: pdata
        end subroutine save_image_c

        function get_wall_time_c() bind(c, name="get_wall_time_c")
            import :: c_double
            real(c_double) :: get_wall_time_c
        end function get_wall_time_c
    end interface

    character(len=256) :: arg_tech, arg_input, arg_output
    integer :: technique, num_args
    integer(c_int) :: width, height, channels
    type(c_ptr) :: c_in_data, c_out_data
    integer(c_int8_t), pointer :: in_arr(:,:,:) => null()
    integer(c_int8_t), pointer :: out_arr(:,:,:) => null()
    integer(c_size_t) :: total_bytes
    real(c_double) :: t_start, t_end, elapsed
    integer :: x, y, c, ky, kx, px, py
    real(c_float) :: sum_val
    real(c_float), parameter :: k(3, 3) = reshape([ &
        1.0/16.0, 2.0/16.0, 1.0/16.0, &
        2.0/16.0, 4.0/16.0, 2.0/16.0, &
        1.0/16.0, 2.0/16.0, 1.0/16.0 ], [3, 3])

    num_args = command_argument_count()
    if (num_args < 3) then
        write(*,*) "Usage: fortran_conv <technique 1-4> <input_image> <output_image>"
        stop 1
    end if

    call get_command_argument(1, arg_tech)
    call get_command_argument(2, arg_input)
    call get_command_argument(3, arg_output)
    read(arg_tech, *) technique

    c_in_data = load_image_c(trim(arg_input) // c_null_char, width, height, channels)
    if (.not. c_associated(c_in_data)) then
        write(*,*) "Failed to load image: ", trim(arg_input)
        stop 1
    end if

    total_bytes = int(width, c_size_t) * int(height, c_size_t) * int(channels, c_size_t)
    allocate(out_arr(channels, width, height))

    ! Bind C pointer to 3D Fortran array: channels x width x height (column-major)
    call c_f_pointer(c_in_data, in_arr, [channels, width, height])

    t_start = get_wall_time_c()

    if (technique == 3) then
        ! Row-based parallelization (horizontal strips)
        !$omp parallel do private(x, c, ky, kx, py, px, sum_val) schedule(static)
        do y = 1, height
            do x = 1, width
                do c = 1, channels
                    sum_val = 0.0
                    do ky = -1, 1
                        py = min(max(y + ky, 1), height)
                        do kx = -1, 1
                            px = min(max(x + kx, 1), width)
                            ! Convert int8 to unsigned 0-255 float
                            sum_val = sum_val + real(iand(int(in_arr(c, px, py)), 255)) * k(kx + 2, ky + 2)
                        end do
                    end do
                    sum_val = min(max(sum_val + 0.5, 0.0), 255.0)
                    out_arr(c, x, y) = int(sum_val, c_int8_t)
                end do
            end do
        end do
        !$omp end parallel do
    else
        ! General parallel loop across rows
        !$omp parallel do private(x, c, ky, kx, py, px, sum_val) schedule(guided)
        do y = 1, height
            do x = 1, width
                do c = 1, channels
                    sum_val = 0.0
                    do ky = -1, 1
                        py = min(max(y + ky, 1), height)
                        do kx = -1, 1
                            px = min(max(x + kx, 1), width)
                            sum_val = sum_val + real(iand(int(in_arr(c, px, py)), 255)) * k(kx + 2, ky + 2)
                        end do
                    end do
                    sum_val = min(max(sum_val + 0.5, 0.0), 255.0)
                    out_arr(c, x, y) = int(sum_val, c_int8_t)
                end do
            end do
        end do
        !$omp end parallel do
    end if

    t_end = get_wall_time_c()
    elapsed = t_end - t_start

    if (trim(arg_output) /= "none") then
        call save_image_c(trim(arg_output) // c_null_char, width, height, channels, c_loc(out_arr))
    end if

    call free_image_c(c_in_data)
    deallocate(out_arr)

    write(*, '(A, F10.6)') "EXECUTION_TIME: ", elapsed
end program convolution_fortran
