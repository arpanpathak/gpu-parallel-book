"""The figures of chapter 12, drawn with figkit and figparts.

    python3 tools/draw_ch12.py

The sizes and times are the ones measured with code/ch12_nvrtc/nvrtc.cu on a Jetson Orin NX
(the runs quoted in the chapter).
"""
import math

from figkit import sheet, width_of
from figparts import (EDGE, INK, KINDS, MUTED, WIRE, arrow, card, line, mono, note, save, tag, text,
                     wire, zone)

OUT = "src/figures/"


def box(x, y, w, h, kind, label="", on=True, cls="mono"):
    """A cell: tinted with an accent edge when `on`, white with a grey edge when not."""
    accent, tint = KINDS[kind]
    parts = [f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
             f'fill="{tint if on else "#FFFFFF"}" stroke="{accent if on else EDGE}" stroke-width="1.2"/>']
    if label:
        parts.append(text(x + w / 2, y + h / 2 + 4, label, cls, "middle", accent if on else MUTED))
    return "\n".join(parts)


def harrow(x1, x2, y, kind="flow"):
    """A straight horizontal arrow from x1 to x2, its tip 3 units short of x2."""
    colour = WIRE if kind == "flow" else KINDS[kind][0]
    marker = "head" if kind == "flow" else f"head-{kind}"
    step = -3 if x2 > x1 else 3
    return (f'  <path d="M{x1},{y} H{x2 + step}" fill="none" stroke="{colour}" stroke-width="1.8"'
            f' stroke-linecap="round" marker-end="url(#{marker})"/>')


# 12.1  The layers between a program and the GPU.
parts = [card(24, 96, 196, 56, "client", "C++, <<<...>>>", "Chapters 3 to 11"),
         card(232, 96, 196, 56, "outside", "C++ at run time", "this chapter"),
         card(440, 96, 196, 56, "store", "Rust, cuda-oxide", "Chapter 13 on"),
         card(24, 184, 196, 84, "client", "runtime API", ["context on first call", "kernels registered", "<<<>>> packs arguments"], pill="cuda*"),
         card(232, 184, 196, 84, "outside", "NVRTC", ["source text in", "PTX or cubin out", "no GPU needed"], pill="nvrtc*"),
         card(440, 184, 196, 84, "store", "cuda-core", ["CudaContext", "CudaModule", "typed launches"]),
         card(24, 300, 612, 64, "service", "driver API, libcuda", "contexts, modules, functions, memory, streams, cuLaunchKernel", pill="cu*"),
         card(24, 396, 300, 52, "idle", "kernel-mode driver", "in the operating system"),
         card(336, 396, 300, 52, "good", "GPU", "runs the machine code")]
parts += [arrow(122, 152, 184), arrow(330, 152, 184), arrow(538, 152, 184),
          arrow(122, 268, 300), arrow(330, 268, 300), arrow(538, 268, 300),
          arrow(174, 364, 396), arrow(486, 364, 396),
          note(24, 478, "The runtime API and cuda-core are both built on the driver API. NVRTC is a compiler: it", room=612),
          note(24, 496, "turns text into PTX or a cubin, and the program hands the result to the driver API.", room=612)]
save(OUT + "ch12-layers.svg", 660, 516,
     "4 rows of cards. Top: 3 kinds of program, C++ with triple angle brackets, C++ built at run time, and Rust "
     "with cuda-oxide. Below them: the runtime API, which creates the context on the first call, registers kernels "
     "and packs arguments; NVRTC, which turns source text into PTX or a cubin; and cuda-core. All 3 sit on the "
     "driver API in libcuda, which reaches the kernel-mode driver and the GPU.",
     "From a program to the GPU", "3 ways in, 1 driver API underneath.", parts)


# 12.2  The compilation pipeline: at build time with nvcc, and while the program runs with NVRTC.
parts = [zone(16, 100, 628, 140, "at build time: nvcc"), zone(16, 272, 628, 216, "while the program runs: NVRTC")]
parts += [card(32, 132, 132, 84, "client", "source", ["CUDA C++", ".cu file"]),
          card(188, 132, 132, 84, "outside", "PTX", ["virtual GPU", "compute_87"]),
          card(344, 132, 132, 84, "good", "cubin", ["machine code", "sm_87 only"]),
          card(500, 132, 128, 84, "store", "fatbin", ["cubins + PTX", "in the binary"]),
          harrow(164, 188, 174), harrow(320, 344, 174), harrow(476, 500, 174)]
parts += [card(32, 304, 160, 84, "client", "source string", ["made by the program", "1,315 bytes"]),
          card(250, 304, 170, 84, "outside", "PTX", ["2,237 bytes", "compile 44 to 88 ms"]),
          card(250, 400, 170, 72, "good", "cubin, sm_87", ["3,624 bytes", "compile 49 to 74 ms"]),
          card(478, 304, 150, 84, "outside", "load PTX", ["driver compiles", "0.7 to 45 ms"]),
          card(478, 400, 150, 72, "good", "load cubin", ["ready to run", "0.05 ms"]),
          harrow(192, 250, 346), wire([(112, 388), (112, 436), (250, 436)]),
          harrow(420, 478, 346), harrow(420, 478, 436)]
save(OUT + "ch12-pipeline.svg", 660, 508,
     "Two zones. At build time, nvcc turns a .cu source file into PTX for a virtual GPU, compute_87, then into a "
     "cubin of machine code for sm_87 only, and bundles cubins and PTX into a fatbin inside the executable. While "
     "the program runs, NVRTC turns a source string of 1,315 bytes into PTX of 2,237 bytes in 44 to 88 ms or a "
     "cubin of 3,624 bytes in 49 to 74 ms. Loading the PTX makes the driver compile it, 0.7 to 45 ms; loading the "
     "cubin takes 0.05 ms.",
     "Source, PTX, cubin", "The same 3 forms, made before the program starts or while it runs.", parts)


# 12.3  Horner's rule worked by hand for degree 2.
parts = [zone(16, 100, 628, 300, "worked by hand"),
         note(36, 140, "p(x) = 1 + x / 2 + x² / 3, so c[0] = 1, c[1] = 1/2, c[2] = 1/3; take x = 0.5.", room=592)]
steps = [("start", "p = c[2]", "1/3", "0.3333"),
         ("k = 1", "p = p × x + c[1]", "0.3333 × 0.5 + 0.5", "0.6667"),
         ("k = 0", "p = p × x + c[0]", "0.6667 × 0.5 + 1", "1.3333")]
for row, (name, rule, work, value) in enumerate(steps):
    y = 160 + 56 * row
    parts += [tag(36, y + 13, name, "idle"),
              box(110, y, 170, 40, "outside", rule),
              box(296, y, 190, 40, "client", work),
              box(502, y, 120, 40, "good", "p = " + value)]
parts += [note(36, 340, "Check, term by term: 1 + 0.5 / 2 + 0.25 / 3 = 1 + 0.25 + 0.0833 = 1.3333.", room=592),
          note(36, 364, "A polynomial of degree d takes d multiply-adds, 1 per coefficient after the first,", room=592),
          note(36, 382, "and no powers of x. Each step needs the result of the step before.", room=592)]
save(OUT + "ch12-horner.svg", 660, 420,
     "Horner's rule for p of x = 1 + x over 2 + x squared over 3 at x = 0.5, in 3 rows. Start: p = c[2] = 1/3 = "
     "0.3333. k = 1: p = 0.3333 times 0.5 plus 0.5 = 0.6667. k = 0: p = 0.6667 times 0.5 plus 1 = 1.3333, which "
     "matches 1 + 0.25 + 0.0833 computed term by term.",
     "Horner's rule by hand", "Degree 2: 2 multiply-adds, from the highest coefficient down.", parts)


# 12.4  The generic kernel and the specialised one, side by side, with their times.
generic = [("", "float p = c[degree];"), ("", "for (k = degree - 1;"), ("", "     k >= 0; --k)"),
           ("", "  p = p * v + c[k];")]
special = [("", "float p = 3.03e-2f;"), ("", "p = p * v + 3.13e-2f;"), ("", "p = p * v + 3.23e-2f;"),
           ("", "... 29 more lines ..."), ("", "p = p * v + 1.00e+0f;")]
parts = [sheet(24, 96, 296, 196, "client", "generic: built by nvcc", generic,
               ["per step: read c[k], count k,", "then 1 multiply-add"]),
         sheet(340, 96, 296, 196, "good", "specialized: built by NVRTC", special,
               ["per step: 1 multiply-add,", "the constant inside it"])]
scale = 230 / 2.7
for row, (name, ms, kind) in enumerate([("generic", 2.61, "client"), ("specialised", 1.84, "good"),
                                        ("rated memory, 102 GB/s", 1.32, "idle")]):
    y = 332 + 40 * row
    accent, tint = KINDS[kind]
    dash = ' stroke-dasharray="5 4"' if kind == "idle" else ""
    parts += [note(24, y + 17, name, room=170),
              f'  <rect x="200" y="{y}" width="{ms * scale:.1f}" height="26" rx="6" fill="{tint}" stroke="{accent}"'
              f' stroke-width="1.3"{dash}/>',
              mono(208 + ms * scale, y + 18, f"{ms:.2f} ms", fill=accent)]
parts += [note(24, 470, "16,777,216 values, degree 32. Both move 134 MB: the specialized", room=612),
          note(24, 488, "kernel at 73 GB/s, 71% of the rated 102 GB/s, and the generic one at 52 GB/s.", room=612)]
save(OUT + "ch12-specialise.svg", 660, 508,
     "Two code sheets. The generic kernel, built by nvcc, loops from c[degree] down, and each step reads c[k] "
     "from memory, counts k and does 1 multiply-add. The specialized kernel, built by NVRTC, is 32 straight lines "
     "such as p = p times v plus 3.13e-2f: each step is 1 multiply-add with the constant inside it. Below, bars: "
     "generic 2.61 ms, specialized 1.84 ms, and the time to move 134 MB at the rated 102 GB/s, 1.32 ms.",
     "Generic and specialized", "The same polynomial; 1 kernel reads its coefficients, the other contains them.",
     parts)


# 12.5  The argument array of cuLaunchKernel, right and wrong.
parts = [zone(16, 100, 628, 236, "args holds the address of each argument"),
         zone(16, 368, 628, 116, "args holds the argument itself")]
names = [("&X", "X", "0x203a00000", "8 bytes"), ("&Y", "Y", "0x207a00000", "8 bytes"),
         ("&n", "n", "16,777,216", "4 bytes")]
parts.append(note(36, 140, "args[ ]", room=80))
for row, (ref, var, value, size) in enumerate(names):
    y = 150 + 52 * row
    parts += [box(36, y, 80, 36, "outside", ref), harrow(116, 196, y + 18),
              box(196, y, 196, 36, "client", f"{var} = {value}"),
              harrow(392, 440, y + 18), box(440, y, 84, 36, "good", size), harrow(524, 540, y + 18)]
parts += [card(540, 150, 92, 140, "service", "kernel", ["x", "y", "n"]),
          note(36, 318, "The driver follows each pointer and copies 8 + 8 + 4 = 20 bytes of parameters.", room=596)]
parts += [box(36, 400, 80, 36, "danger", "X"), harrow(116, 196, 418, "danger"),
          box(196, 400, 196, 36, "danger", "0x203a00000"),
          note(410, 414, "the CPU reads device memory:", room=220),
          note(410, 432, "segmentation fault, exit 139", room=220, fill=KINDS["danger"][0]),
          note(36, 466, "The device address is taken for the address of an argument on the host.", room=596)]
save(OUT + "ch12-args.svg", 660, 504,
     "Top zone, the correct form: args[0] = &X points to the host variable X, which holds the device address "
     "0x203a00000, 8 bytes; args[1] = &Y points to Y, 8 bytes; args[2] = &n points to n = 16,777,216, 4 bytes. "
     "The driver follows each pointer and copies 20 bytes of parameters into the kernel's x, y and n. Bottom "
     "zone, the mistake: args[0] = X, so the driver reads 8 bytes at the device address 0x203a00000 on the CPU, "
     "and the program stops with a segmentation fault, exit code 139.",
     "The argument array", "Each entry is the address of 1 argument, not the argument.", parts)


# 12.6  The JIT cache.
parts = [card(24, 100, 196, 64, "outside", "cuModuleLoadData", "PTX, 2,237 bytes"),
         card(232, 100, 404, 64, "store", "look up the PTX in the cache", "key: the PTX, the GPU, the driver version"),
         harrow(220, 232, 132),
         card(24, 208, 300, 84, "danger", "miss: compile", ["the driver compiles the PTX", "13 to 45 ms, measured"]),
         card(336, 208, 300, 84, "good", "hit: copy", ["machine code from the file", "0.7 ms, measured"]),
         card(24, 324, 300, 64, "store", "store the result", "for the next process"),
         card(24, 420, 612, 56, "service", "module in the context", "the same machine code either way"),
         wire([(434, 164), (434, 186), (174, 186), (174, 208)]),
         arrow(486, 164, 208), arrow(174, 292, 324), arrow(600, 292, 420),
         wire([(174, 388), (174, 404), (330, 404), (330, 420)])]
parts += [tag(250, 177, "first time", "danger"), tag(494, 177, "later", "good"),
          note(352, 344, "where: CUDA_CACHE_PATH,", room=284),
          note(352, 362, "by default ~/.nv/ComputeCache", room=284),
          note(352, 380, "off: CUDA_CACHE_DISABLE=1", room=284)]
save(OUT + "ch12-cache.svg", 660, 496,
     "A flow. cuModuleLoadData receives 2,237 bytes of PTX and looks it up in the cache, keyed by the PTX, the GPU "
     "and the driver version. The first time, a miss: the driver compiles the PTX, 13 to 45 ms when measured, and "
     "stores the result for the next process, in CUDA_CACHE_PATH, by default ~/.nv/ComputeCache. Later, a hit: "
     "the machine code is copied from the file in 0.7 ms. Either way the module in the context holds the same code.",
     "The JIT cache", "A process pays to compile PTX once; later processes read the result.", parts)


# 12.7  1 primary context, used by the runtime API, the driver API and cuda-core.
parts = [card(24, 100, 196, 84, "client", "runtime API", ["cudaMalloc", "cudaMemcpy", "<<<...>>>"]),
         card(24, 212, 196, 84, "service", "driver API", ["cuModuleLoadData", "cuLaunchKernel", "cuModuleUnload"]),
         card(24, 324, 196, 84, "store", "cuda-core", ["CudaContext::new", "load_module_from_image", "launch_kernel_on_stream"]),
         zone(260, 100, 384, 308, "primary context of GPU 0"),
         card(284, 132, 336, 64, "client", "memory", "X, Y and C from cudaMalloc"),
         card(284, 212, 336, 64, "service", "module from the fatbin", "polyGeneric, built by nvcc"),
         card(284, 292, 336, 64, "outside", "module from NVRTC", "poly, built while the program ran"),
         wire([(220, 142), (260, 142)]), wire([(220, 254), (260, 254)]), wire([(220, 366), (260, 366)])]
parts += [note(24, 438, "cuDevicePrimaryCtxRetain hands out the context the runtime API uses, so a pointer", room=612),
          note(24, 456, "from cudaMalloc is valid in cuLaunchKernel. cuda-core retains the same context.", room=612)]
save(OUT + "ch12-context.svg", 660, 476,
     "On the left, 3 callers: the runtime API with cudaMalloc, cudaMemcpy and triple angle brackets; the driver API "
     "with cuModuleLoadData, cuLaunchKernel and cuModuleUnload; and cuda-core with CudaContext::new, "
     "load_module_from_image and launch_kernel_on_stream. On the right, 1 primary context of GPU 0 holds the "
     "memory X, Y and C from cudaMalloc, the module from the fatbin with polyGeneric, and the module from NVRTC "
     "with poly. All 3 callers reach it.",
     "1 context, 3 callers", "The runtime API and the driver API share the primary context.", parts)


# 12.8  Where the milliseconds go, on a log scale, and when compiling pays back.
rows = [("compile to cubin", 52.0, "outside"), ("load PTX, cache miss", 32.0, "danger"),
        ("generic kernel, 1 run", 2.61, "client"), ("specialized kernel, 1 run", 1.84, "good"),
        ("load PTX, cache hit", 0.77, "store"), ("load cubin", 0.05, "service")]
left, right, low, high = 230, 610, math.log10(0.01), math.log10(100)
X = lambda ms: left + (right - left) * (math.log10(ms) - low) / (high - low)
parts = []
for tick in (0.01, 0.1, 1, 10, 100):
    parts += [line(X(tick), 104, X(tick), 352, EDGE, 1, "3 4"),
              mono(X(tick), 370, f"{tick:g} ms", "middle", MUTED)]
for row, (name, ms, kind) in enumerate(rows):
    y = 112 + 40 * row
    accent, tint = KINDS[kind]
    parts += [note(24, y + 17, name, room=200),
              f'  <rect x="{left}" y="{y}" width="{X(ms) - left:.1f}" height="24" rx="6" fill="{tint}"'
              f' stroke="{accent}" stroke-width="1.3"/>',
              mono(X(ms) + 8, y + 17, f"{ms:g} ms", fill=accent)]
parts += [card(24, 396, 612, 84, "good", "when compiling pays back",
               ["each run saves 2.61 − 1.84 = 0.77 ms", "compiling and loading cost 52 ms",
                "52 ms ÷ 0.77 ms = 68 runs to break even"])]
save(OUT + "ch12-costs.svg", 660, 500,
     "Horizontal bars on a logarithmic time axis from 0.01 ms to 100 ms, measured: compile to "
     "cubin 52 ms; load PTX with a cache miss 32 ms; 1 run of the generic kernel 2.61 ms; 1 run of the specialized "
     "kernel 1.84 ms; load PTX with a cache hit 0.77 ms; load cubin 0.05 ms. Below, a card: each run saves 0.77 ms, "
     "compiling costs 52 ms, so it pays back after 68 runs.",
     "Where the milliseconds go", "Compiling costs as much as 20 to 30 runs of the kernel itself.", parts)
