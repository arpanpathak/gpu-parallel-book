"""The figures of chapter 10, drawn with figkit and figparts.

    python3 tools/draw_ch10.py

The numbers are the ones measured with code/ch10_modern/raw.cu and modern.cu on the first
GPU named in the chapter (the runs quoted there).
"""
from figkit import stacked, width_of
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, fan, legend, line, mono, note, save,
                      tag, text, zone)

OUT = "src/figures/"


def box(x, y, w, h, kind, label="", on=True, cls="mono", dashed=False):
    """A rounded box: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.3"{dash}/>']
    if label:
        size = 12 if cls == "mono" else 12.5
        room = w - 8
        if width_of(label, size, mono=cls == "mono") > room:
            raise ValueError(f"{label!r} does not fit in {w}")
        parts.append(text(x + w / 2, y + h / 2 + 4.2, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def gather(xs, y, bus, x, bottom):
    """Several sources joined into 1: a drop from each to the bus line, then 1 arrow down."""
    left, right = min(xs + [x]), max(xs + [x])
    parts = [f'  <path d="M{left},{bus} H{right}" fill="none" stroke="{WIRE}" stroke-width="1.8"/>']
    parts += [f'  <path d="M{sx},{y} V{bus}" fill="none" stroke="{WIRE}" stroke-width="1.6"/>'
              for sx in xs]
    parts += [f'  <circle cx="{x}" cy="{bus}" r="3" fill="{WIRE}"/>', arrow(x, bus, bottom)]
    return "\n".join(parts)


def down(x, y1, y2, kind="flow"):
    return arrow(x, y1, y2, kind)


# 10.1  The early return: who frees the 2 buffers, with raw pointers and with DeviceBuffer.
parts = [zone(16, 104, 304, 376, "raw pointers"), zone(340, 104, 304, 376, "DeviceBuffer")]
steps_raw = [("cudaMallocAsync(&in, 8 MiB)", "client", True, False),
             ("cudaMallocAsync(&out, 8 MiB)", "client", True, False),
             ("if (factor == 0) return false;", "store", True, False),
             ("cudaFreeAsync(in, 0);", "danger", False, True),
             ("cudaFreeAsync(out, 0);", "danger", False, True)]
steps_raii = [("DeviceBuffer<float> in(host);", "client", True, False),
              ("DeviceBuffer<float> out(n);", "client", True, False),
              ("if (factor == 0) return false;", "store", True, False),
              ("~out(): cudaFreeAsync", "good", True, False),
              ("~in():  cudaFreeAsync", "good", True, False)]
for x0, steps in ((40, steps_raw), (364, steps_raii)):
    for k, (label, kind, on, dashed) in enumerate(steps):
        y = 132 + k * 50
        parts.append(box(x0, y, 256, 30, kind, label, on, dashed=dashed))
        if k < 2 or (k >= 2 and steps is steps_raii and k < 4):
            parts.append(down(x0 + 128, y + 30, y + 50))
parts += [tag(40, 380, "skipped on every early return", "danger"),
          tag(364, 380, "run on every way out", "good")]
for x0, lost, kind in ((40, 128, "danger"), (364, 0, "good")):
    parts.append(note(x0, 424, f"pool in use after 8 calls: {lost} MiB", room=256))
    if lost:
        accent = KINDS[kind][0]
        parts.append(stacked(x0, 434, 256, 22, [(16, accent if k % 2 == 0 else "#E36B4F", "16")
                                                  for k in range(8)], total=128))
    else:
        parts.append(f'  <rect x="{x0}" y="434" width="256" height="22" rx="6" fill="#FFFFFF" '
                     f'stroke="{EDGE}" stroke-width="1.2"/>')
        parts.append(text(x0 + 128, 449, "empty", "pill", "middle", MUTED))
parts += [note(16, 508, "Each call takes 2 buffers of 8 MiB from the pool. The raw function returns before", room=628),
          note(16, 526, "its 2 frees, so 8 calls keep 8 × 16 MiB = 128 MiB. The destructors run on the", room=628),
          note(16, 544, "return itself, in the reverse order of construction: out first, then in.", room=628)]
save(OUT + "ch10-leak.svg", 660, 566,
     "Two columns of steps. Raw pointers: 2 calls of cudaMallocAsync of 8 MiB, then if factor is 0 "
     "return false; the 2 calls of cudaFreeAsync below are dashed, skipped on every early return. After 8 "
     "calls the pool holds 8 segments of 16 MiB, 128 MiB. DeviceBuffer: the same 2 allocations and the "
     "same return, followed by the destructors of out and in, which free both buffers; after 8 calls the "
     "pool is empty.",
     "The early return", "Raw pointers skip the frees; a destructor runs on the return itself.", parts)


# 10.2  A count of bytes where a count of floats was meant: N = 16.
parts = [zone(16, 104, 628, 300, "16 floats, copied with a count of 16")]
cw, x0 = 36, 42
parts.append(note(x0, 140, "host: 16 floats of 4 bytes, 64 bytes", room=400))
for k in range(16):
    parts.append(box(x0 + k * cw, 150, cw - 4, 28, "client", "1.0"))
parts += [f'  <path d="M{x0},{196} V{204} H{x0 + 4 * cw - 4} V{196}" fill="none" stroke="{KINDS["good"][0]}" '
          f'stroke-width="1.6"/>',
          mono(x0 + 2 * cw - 2, 222, "16 bytes", "middle", KINDS["good"][0]),
          mono(x0 + 4 * cw + 12, 222, "cudaMemcpy(in, host, 16, ...) copies these, and stops", "start", INK)]
parts.append(note(x0, 256, "device, in: 4 floats arrive; 12 keep the 0.0 of the memset", room=500))
for k in range(16):
    parts.append(box(x0 + k * cw, 266, cw - 4, 28, "good" if k < 4 else "danger", "1.0" if k < 4 else "0.0"))
parts.append(note(x0, 322, "device, out = 2 × in", room=400))
for k in range(16):
    parts.append(box(x0 + k * cw, 332, cw - 4, 28, "good" if k < 4 else "danger", "2.0" if k < 4 else "0.0"))
parts += [tag(x0, 372, "4 of 16 correct: 16 bytes ÷ 4 bytes per float = 4 floats", "danger"),
          note(16, 432, "In raw.cu, N = 2,097,152: 2,097,152 bytes are 524,288 floats, so 524,288 of", room=628),
          note(16, 450, "2,097,152 results are right, 1 in 4. No call returns an error: the copy did what", room=628),
          note(16, 468, "its arguments asked for.", room=628)]
save(OUT + "ch10-bytes.svg", 660, 490,
     "A row of 16 host floats, all 1.0. A bracket over the first 4 marks 16 bytes: cudaMemcpy with a count "
     "of 16 copies only those. On the device, the first 4 floats of in are 1.0 and the other 12 are 0.0; "
     "after the kernel doubles them, out holds 4 values of 2.0 and 12 of 0.0. 4 of 16 are correct, because "
     "16 bytes are 4 floats. In the program, 524,288 of 2,097,152.",
     "Bytes, not elements", "cudaMemcpy counts bytes. Given a count of floats, it copies 1/4 of them.", parts)


# 10.3  The lifetime of a DeviceBuffer: the destructor runs on every way out of the scope.
cx = 250
parts = [zone(16, 104, 470, 420, "1 scope"),
         box(cx - 160, 130, 320, 32, "client", "DeviceBuffer<float> in(host);"),
         down(cx, 162, 184),
         box(cx - 160, 184, 320, 32, "good", "constructor: cudaMallocAsync, copy"),
         down(cx, 216, 238),
         box(cx - 160, 238, 320, 32, "idle", "use: kernels read in.view()"),
         fan(cx, 270, [cx - 150, cx, cx + 150], 292, 314)]
exits = [("return false;", "store"), ("check() throws", "danger"), ("closing }", "idle")]
for k, (label, kind) in enumerate(exits):
    parts.append(box(cx - 150 + 150 * k - 66, 314, 132, 32, kind, label))
parts += [gather([cx - 150, cx, cx + 150], 346, 368, cx, 390),
          box(cx - 160, 390, 320, 32, "good", "destructor: cudaFreeAsync"),
          down(cx, 422, 444),
          box(cx - 160, 444, 320, 32, "service", "8 MiB back in the pool"),
          note(504, 150, "The compiler places", room=140), note(504, 168, "the destructor call", room=140),
          note(504, 186, "on every path that", room=140), note(504, 204, "leaves the scope:", room=140),
          note(504, 222, "a return, an", room=140), note(504, 240, "exception passing", room=140),
          note(504, 258, "through, or the end", room=140), note(504, 276, "of the block.", room=140),
          note(504, 312, "Nothing in the", room=140), note(504, 330, "function body asks", room=140),
          note(504, 348, "for it, so nothing", room=140), note(504, 366, "can forget it.", room=140)]
save(OUT + "ch10-lifetime.svg", 660, 546,
     "A column of steps inside a zone labeled 1 scope. DeviceBuffer<float> in(host) calls the constructor, "
     "which allocates with cudaMallocAsync and copies; kernels then read in.view(). 3 ways out branch from "
     "there: return false, check() throws, and the closing brace. All 3 join into 1 box, the destructor, "
     "which calls cudaFreeAsync and gives the 8 MiB back to the pool.",
     "The lifetime of a DeviceBuffer", "3 ways out of a scope, 1 destructor.", parts)


# 10.4  Copy versus move: 2 owners of 1 allocation, or 1 owner after a transfer.
parts = [zone(16, 104, 304, 372, "copy: deleted"), zone(340, 104, 304, 372, "move: allowed")]
for x0, a_ptr, kinds in ((40, "0x7f00", ("danger", "danger")), (364, "nullptr", ("idle", "good"))):
    parts += [box(x0, 136, 116, 52, kinds[0], ""), box(x0 + 140, 136, 116, 52, kinds[1], ""),
              text(x0 + 58, 156, "a", "name", "middle"), mono(x0 + 58, 176, f"data_ = {a_ptr}", "middle", INK),
              text(x0 + 198, 156, "b", "name", "middle"), mono(x0 + 198, 176, "data_ = 0x7f00", "middle", INK)]
    if a_ptr != "nullptr":
        parts.append(down(x0 + 58, 188, 250, "danger"))
    parts += [down(x0 + 198, 188, 250, "danger" if a_ptr != "nullptr" else "flow"),
              box(x0, 250, 256, 40, "store", "8 MiB at 0x7f00")]
parts += [mono(40, 226, "b = a", "start", MUTED), mono(364, 226, "b = std::move(a)", "start", MUTED),
          tag(40, 312, "~b frees 0x7f00", "danger"), tag(40, 340, "~a frees 0x7f00 again", "danger"),
          tag(364, 312, "~b frees 0x7f00", "good"), tag(364, 340, "~a: nullptr, nothing to free", "idle"),
          note(40, 388, "2 owners free 1 allocation", room=256), note(40, 406, "twice. The deleted copy", room=256),
          note(40, 424, "turns b = a into a compile", room=256), note(40, 442, "error.", room=256),
          note(364, 388, "The move hands the pointer", room=256), note(364, 406, "to b and leaves a empty.", room=256),
          note(364, 424, "1 owner, 1 free; no byte", room=256), note(364, 442, "of the 8 MiB is copied.", room=256)]
save(OUT + "ch10-ownership.svg", 660, 498,
     "Two panels. Copy, deleted: objects a and b both hold data_ = 0x7f00 and both point at the same 8 MiB "
     "allocation; ~b frees it and ~a frees it again, a double free, so the copy is made a compile error. "
     "Move, allowed: after b = std::move(a), b holds 0x7f00 and a holds nullptr; only b points at the "
     "allocation, ~b frees it once and ~a has nothing to free.",
     "Copy or move", "A copied owner frees twice; a moved owner hands its pointer over.", parts)


# 10.5  1 template, 3 kernels: what the compiler instantiates, and the measured rates.
parts = [box(150, 104, 360, 30, "outside", "template <Number T, UnaryOp<T> F>"),
         box(150, 138, 360, 30, "outside", "__global__ void transform(in, out, f)"),
         fan(330, 168, [120, 330, 540], 196, 220)]
kernels = [("transform<float>", "4 bytes in, 4 bytes out", "74.6 GB/s"),
           ("transform<double>", "8 bytes in, 8 bytes out", "90.2 GB/s"),
           ("transform<int>", "4 bytes in, 4 bytes out", "77.0 GB/s")]
for k, (name, sub, rate) in enumerate(kernels):
    x = 120 + 210 * k - 96
    parts += [box(x, 220, 192, 94, "client", ""),
              text(x + 96, 246, name, "mono strong", "middle", KINDS["client"][0]),
              text(x + 96, 268, sub, "sub", "middle"),
              tag(x + 96 - width_of(rate, 11, bold=True) / 2 - 7, 284, rate, "good")]
parts += [box(126, 344, 408, 30, "idle", "scaleRaw, written by hand for float: 74.5 GB/s"),
          note(16, 408, "The template is not a kernel. At each launch with new types, the compiler writes", room=628),
          note(16, 426, "a kernel for those types, as if typed by hand: 3 types, 3 kernels in the binary.", room=628),
          note(16, 444, "transform<float> ran at the speed of the hand-written kernel (16,777,216 values).", room=628),
          note(16, 462, "double moves twice the bytes per thread and reached 90.2 GB/s of 102 GB/s.", room=628)]
save(OUT + "ch10-instantiate.svg", 660, 484,
     "A template, transform with type T and operation F, fans out into 3 kernels the compiler writes: "
     "transform<float>, 4 bytes in and 4 out, 74.6 GB/s; transform<double>, 8 bytes in and 8 out, 90.2 GB/s; "
     "transform<int>, 4 bytes in and 4 out, 77.0 GB/s. Below, the hand-written float kernel scaleRaw ran at "
     "74.5 GB/s.",
     "1 template, 3 kernels", "The compiler writes 1 kernel per set of types; each runs at hand-written speed.", parts)


# 10.6  1 rounding or 2: why the same expression gives different bits on the CPU and the GPU.
parts = [zone(16, 104, 304, 300, "multiply, then add: 2 roundings"), zone(340, 104, 304, 300, "fused: 1 rounding")]
left = [("1.23 × 4.56 = 5.6088", "idle"), ("round to 3 digits: 5.61", "store"),
        ("5.61 − 5.60 = 0.0100", "danger")]
right = [("1.23 × 4.56 − 5.60", "idle"), ("exact: 0.0088", "idle"),
         ("round to 3 digits: 0.00880", "good")]
for x0, steps in ((40, left), (364, right)):
    for k, (label, kind) in enumerate(steps):
        y = 140 + k * 66
        parts.append(box(x0, y, 256, 34, kind, label))
        if k < 2:
            parts.append(down(x0 + 128, y + 34, y + 66))
parts += [note(40, 366, "the product is rounded before the add", room=256),
          note(364, 366, "the product stays exact inside the add", room=256),
          note(16, 436, "With 3 significant digits, the 2 results differ by 14%. A float keeps 24 binary digits,", room=628),
          note(16, 454, "so the difference is in the last bit, but it is there. The GPU compiler fuses a × b + c", room=628),
          note(16, 472, "into 1 FMA instruction; the host compiler did not. expPoly: 585,903 of 2,097,152", room=628),
          note(16, 490, "values differ between CPU and GPU. expPolyFma, with fmaf written out: 0 differ.", room=628)]
save(OUT + "ch10-fma.svg", 660, 512,
     "Two columns worked with 3 significant decimal digits. Left, 2 roundings: 1.23 times 4.56 = 5.6088, "
     "rounded to 5.61, minus 5.60 gives 0.0100. Right, 1 rounding: 1.23 times 4.56 minus 5.60 is exactly "
     "0.0088, rounded to 0.00880. In the program, the polynomial written as x * y + z differs between CPU "
     "and GPU in 585,903 of 2,097,152 values; written with fmaf on both sides, in 0.",
     "1 rounding or 2", "A fused multiply-add rounds once; a multiply then an add rounds twice.", parts)


# 10.7  The scopes of cuda::atomic_ref: which threads must see the update.
parts = [zone(16, 104, 628, 360, "system: the CPU and every GPU"),
         zone(40, 148, 580, 236, "device: every thread of 1 GPU"),
         zone(64, 192, 260, 168, "block: 256 threads of 1 block"),
         box(88, 230, 212, 30, "idle", "thread_scope_block", cls="mono"),
         note(88, 290, "shared-memory counters of", room=212), note(88, 308, "1 block (Chapter 5)", room=212),
         box(348, 196, 248, 30, "client", "thread_scope_device"),
         note(348, 250, "countAbove: 8,192 blocks of", room=248), note(348, 268, "256 threads add to 1 int", room=248),
         note(348, 286, "in global memory", room=248),
         tag(348, 312, "2,097,152 values, 622,552 counted", "good"),
         box(64, 408, 236, 30, "outside", "thread_scope_system"),
         note(324, 420, "memory that the CPU and the GPU", room=300),
         note(324, 438, "both update, such as a flag", room=300)]
parts += [note(16, 492, "A wider scope promises that more threads see the update, and costs more to keep:", room=628),
          note(16, 510, "use the narrowest scope that contains every thread that touches the value.", room=628)]
save(OUT + "ch10-scopes.svg", 660, 532,
     "Three nested zones. The outer zone, system, holds the CPU and every GPU and goes with "
     "thread_scope_system, for memory that the CPU and the GPU both update, such as a flag. Inside it, device holds every thread of 1 GPU and goes with thread_scope_device: "
     "countAbove runs 8,192 blocks of 256 threads that add to 1 int in global memory, and counted 622,552 of "
     "2,097,152 values. Inside that, block holds the 256 threads of 1 block and goes with thread_scope_block, "
     "as for the shared-memory counters of Chapter 5.",
     "The scope of an atomic", "Each scope names the threads that must agree on the value.", parts)


# 10.8  When each mistake is caught: with the C API, with modern C++, with Rust.
rows = [("leak on early return", ("128 MiB lost", "danger"), ("cannot happen", "service"), ("cannot happen", "service")),
        ("2,048 threads per block", ("silent: out = 0", "danger"), ("throws, file:line", "store"), ("Err from prepare", "store")),
        ("bytes for elements", ("1 in 4 copied", "danger"), ("cannot happen", "service"), ("cannot happen", "service")),
        ("copy of an owner", ("double free", "danger"), ("compile error", "good"), ("moved, not copied", "good")),
        ("block not × 32", ("not checked", "danger"), ("static_assert", "good"), ("const assert!", "good")),
        ("op returns bool", ("no generics", "idle"), ("concept: error", "good"), ("trait bound: error", "good"))]
cols = ["C API", "modern C++", "Rust"]
parts = []
for c, name in enumerate(cols):
    parts.append(text(196 + 148 * c + 70, 112, name, "name", "middle"))
for r, (label, *cells) in enumerate(rows):
    y = 128 + r * 46
    parts.append(text(24, y + 22, label, "sub"))
    for c, (body, kind) in enumerate(cells):
        parts.append(box(196 + 148 * c, y, 140, 34, kind, body, cls="sub"))
parts += [legend(24, 428, [("danger", "never caught"), ("store", "at run time"), ("good", "at compile time"),
                           ("service", "cannot be written")]),
          note(24, 466, "Each row is a mistake run in this chapter. The C API catches none of them. Modern C++", room=612),
          note(24, 484, "and Rust move most of them to compile time, and make the leak and the byte count", room=612),
          note(24, 502, "impossible to write.", room=612)]
save(OUT + "ch10-caught.svg", 660, 524,
     "A table of 6 mistakes against 3 ways of writing the program. Leak on early return: C API 128 MiB lost; "
     "modern C++ and Rust cannot happen. 2,048 threads per block: C API silent, out = 0; C++ throws with file "
     "and line; Rust returns an Err from prepare. Bytes for elements: C API 1 in 4 copied; C++ and Rust cannot "
     "happen. Copy of an owner: C API double free; C++ compile error; Rust moved, not copied. Block not a "
     "multiple of 32: C API not checked; C++ static_assert; Rust const assert. An operation that returns bool: "
     "C API has no generics; C++ concept error; Rust trait bound error.",
     "When each mistake is caught", "6 mistakes of this chapter, written 3 ways.", parts)
