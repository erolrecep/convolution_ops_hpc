from std.sys import argv
from std.python import Python
from std.memory import Pointer
from std.memory.alloc import Layout, alloc

def min_val(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b

def max_val(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b

def main() raises:
    var sys = Python.import_module("sys")
    var os = Python.import_module("os")
    var ctypes = Python.import_module("ctypes")
    var builtins = Python.import_module("builtins")

    var args = argv()
    if len(args) < 4:
        print("Usage: mojo_conv <technique 1-4> <input_image> <output_image>")
        return

    var technique = Int(args[1])
    var input_path = builtins.str(args[2])
    var output_path = builtins.str(args[3])

    var base_dir = os.path.dirname(os.path.abspath(builtins.str(args[0])))
    var dylib_name = "libc_loader.dylib" if sys.platform == "darwin" else "libc_loader.so"
    var cand1 = os.path.join(base_dir, dylib_name)
    var cand2 = os.path.join(base_dir, "..", "src", dylib_name)
    var cand3 = os.path.join(base_dir, "src", dylib_name)
    var lib_path = cand1
    if os.path.exists(cand2):
        lib_path = cand2
    elif os.path.exists(cand3):
        lib_path = cand3

    var lib = ctypes.CDLL(lib_path)
    lib.c_get_time.restype = ctypes.c_double

    var w = ctypes.c_int()
    var h = ctypes.c_int()
    var ch = ctypes.c_int()

    var c_in_path = input_path.encode("utf-8")
    var ok = lib.c_get_image_dims(c_in_path, ctypes.byref(w), ctypes.byref(h), ctypes.byref(ch))
    if not ok:
        print("Failed to query image:", input_path)
        return

    var width = Int(String(w.value))
    var height = Int(String(h.value))
    var channels = Int(String(ch.value))
    var total_bytes = width * height * channels
    var row_len = width * channels

    var in_layout = Layout[UInt8](count=total_bytes)
    var in_alloc = alloc(in_layout)
    var in_ptr = (in_alloc^).unsafe_leak()

    var tmp_layout = Layout[UInt8](count=total_bytes)
    var tmp_alloc = alloc(tmp_layout)
    var tmp_ptr = (tmp_alloc^).unsafe_leak()

    var out_layout = Layout[UInt8](count=total_bytes)
    var out_alloc = alloc(out_layout)
    var out_ptr = (out_alloc^).unsafe_leak()

    var addr_str = String(in_ptr)
    var addr_int = builtins.int(addr_str, 16)
    var in_c_ptr = ctypes.c_void_p(addr_int)

    var loaded = lib.c_load_image_to_ptr(c_in_path, in_c_ptr)
    if not loaded:
        print("Failed to load image pixels:", input_path)
        return

    var t0 = lib.c_get_time()

    # Pass 1: 1D Horizontal Blur
    for y in range(height):
        var y_off = y * row_len
        for x in range(width):
            var x0 = max_val(x - 1, 0)
            var x1 = x
            var x2 = min_val(x + 1, width - 1)
            for c in range(channels):
                var p0 = Int(in_ptr.unsafe_offset(y_off + x0 * channels + c)[])
                var p1 = Int(in_ptr.unsafe_offset(y_off + x1 * channels + c)[])
                var p2 = Int(in_ptr.unsafe_offset(y_off + x2 * channels + c)[])
                var r = (p0 + (p1 << 1) + p2 + 2) >> 2
                tmp_ptr.unsafe_offset(y_off + x * channels + c).unsafe_write(UInt8(r))

    # Pass 2: 1D Vertical Blur
    for y in range(height):
        var y0 = max_val(y - 1, 0)
        var y1 = y
        var y2 = min_val(y + 1, height - 1)
        var y0_off = y0 * row_len
        var y1_off = y1 * row_len
        var y2_off = y2 * row_len
        var out_y_off = y * row_len
        for x in range(width):
            var x_off = x * channels
            for c in range(channels):
                var r0 = Int(tmp_ptr.unsafe_offset(y0_off + x_off + c)[])
                var r1 = Int(tmp_ptr.unsafe_offset(y1_off + x_off + c)[])
                var r2 = Int(tmp_ptr.unsafe_offset(y2_off + x_off + c)[])
                var res = (r0 + (r1 << 1) + r2 + 2) >> 2
                out_ptr.unsafe_offset(out_y_off + x_off + c).unsafe_write(UInt8(res))

    var elapsed = lib.c_get_time() - t0

    if output_path != "none":
        var out_addr_str = String(out_ptr)
        var out_addr_int = builtins.int(out_addr_str, 16)
        var out_c_ptr = ctypes.c_void_p(out_addr_int)
        var c_out_path = output_path.encode("utf-8")
        lib.c_save_image_from_ptr(c_out_path, w.value, h.value, ch.value, out_c_ptr)

    in_ptr.unsafe_free()
    tmp_ptr.unsafe_free()
    out_ptr.unsafe_free()

    print("EXECUTION_TIME:", elapsed)
