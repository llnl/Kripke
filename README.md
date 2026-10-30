KRIPKE
======

Kripke is a simple, scalable, 3D Sn deterministic particle transport code. Its primary purpose is to research how data layout, programming paradigms and architectures effect the implementation and performance of Sn transport. A main goal of Kripke is to investigate how different data-layouts affect instruction, thread and task level parallelism, and what the implications are on overall solver performance.

Kripke supports storage of angular fluxes (Psi) using all six striding orders (or "nestings") of Directions (D), Groups (G), and Zones (Z), and provides computational kernels specifically written for each of these nestings. Most Sn transport codes are designed around one of these nestings, which is an inflexibility that leads to software engineering compromises when porting to new architectures and programming paradigms.

Early research has found that the problem dimensions (zones, groups, directions, scattering order) and the scaling (number of threads and MPI tasks), can make a profound difference in the performance of each of these nestings. To our knowledge, this is a capability unique to Kripke, and should provide key insight into how data-layout affects Sn solver performance. An asynchronous MPI-based parallel sweep algorithm is provided, which employs the concepts of Group Sets (GS), Zone Sets (ZS), and Direction Sets (DS), borrowed from the [Texas A&M code PDT](https://parasol.tamu.edu/asci/).

As we explore new architectures and programming paradigms with Kripke, we will be able to incorporate these findings and ideas into our larger codes. The main advantages of using Kripke for this exploration is that it's light-weight (i.e. easily refactored and modified), and it gets us closer to the real question we want answered: "What is the best way to layout and traverse data in parallel in an Sn code on a given architecture+programming-model?" instead of the more commonly asked question "What is the best way to map my existing Sn code to a given architecture+programming-model?".


Mini App or Proxy App?
----------------------
Kripke is a Mini-App since it has a very small code base consisting of about 5000 lines of C++ code (using cloc v1.67).

Kripke is also a Proxy-App since it is a proxy for the LLNL transport code ARDRA.


Analysis
--------
A major challenge of achieving high-performance in an Sn transport (or any physics) code is choosing a data-layout and a parallel decomposition that lends itself to the targeted architecture. Often the data-layout determines the most efficient nesting of loops in computational kernels, which then determines how well your inner-most-loop SIMD vectorizes, how you add threading (pthreads, OpenMP, etc.), and the efficiency and design of your parallel algorithms. Therefore, each nesting produces a different loop nesting orders with substantially different performance characteristics. We want to explore how easily and efficiently these different nestings map to different architectures. In particular, we are interested in exploring how we can achieve good parallel efficiency while also achieving efficient use of node resources (such as SIMD units, memory systems, and accelerators).

Parallel sweep algorithms can be explored with Kripke in multiple ways. The core MPI algorithm could be modified or rewritten to explore other approaches, domain overloading, or alternate programming models (such as Charm++). The effect of load-imbalance is an understudied aspect of Sn transport sweeps, and could easily be studied with Kripke by artificially adding more work (i.e. unknowns) to a subset of MPI tasks. Block-AMR could be added to Kripke, which would be a useful way to explore the cost-benefit analysis of adding AMR to an Sn code, and would be a way to further study load imbalances and AMR effects on sweeps.

The coupling of the on-node sweep kernel, the parallel sweep algorithm, and the choices of decomposing the problem phase space into GS's, ZS's and DS's impact the performance of the overall sweep. The trade off between large and small "units of work" can be studied. Larger "units of work" provide more opportunity for on-node parallelism, while creating larger messages, less "sends", and less efficient parallel sweeps. Smaller "units of work" make for less efficient on-node kernels, but more efficient parallel sweeps. 

We can also study trading MPI tasks for threads, and the effects this has on our programming models and cache efficiency.

A simple timer infrastructure is provided that measure each compute kernels total time.


Physical Models
---------------

Kripke solves the Discrete Ordinance and Diamond Difference discretized steady-state linear Boltzmann equation. 

        H * Psi = (LPlus * S * L) * Psi + Q

Where:

*   **Psi** is the unknown angular flux discretized over zones, directions, and energy groups

*   **H** is the "streaming-collision" operator.  (Couples zones)

*   **L** is the "discrete-to-moments operator. (Couples directions and moments)

*   **LPlus** is the "moment-to-discrete" operator. (Couples directions and moments)

*   **S** is the (arbitrary) order scattering operator. (Couples groups)

*   **Q** is an external source. In Kripke it is represented in moment space, so really "LPlus*Q"


Kripke is hard-coded to setup and solve the [3D Kobayashi radiation benchmark, problem 3i](https://www.oecd-nea.org/science/docs/2000/nsc-doc2000-4.pdf).  Since Kripke does not have reflecting boundary conditions, the full-space model is solved. Command line arguments allow the user to modify the total and scattering cross-sections.  Since Kripke is a multi-group transport code and the Kobayashi problem is single-group, each energy group is setup to solve the same problem with no group-to-group coupling in the data.


The steady-state solution method uses the source-iteration technique, where each iteration is as follows:

1.  Phi = LTimes(Psi)
2.  PhiOut = Scattering(Phi)
3.  PhiOut = PhiOut + Source()
4.  Rhs = LPlusTimes(PhiOut)
5.  Psi = Sweep(Rhs, Psi)  which is solving Psi=(Hinverse * Rhs) a.k.a _"Inverting H"_



Building and Running
====================

Kripke uses a BLT/CMake superbuild to build the application and its dependencies.

Requirements
------------

Basic requirements:

*  CMake 3.24 or later for the bundled dependencies. CUDA 13 with C++20 requires CMake 3.25 or later.

*  A compiler with C++20 support.

*  (Optional) MPI 1.0 or later

*  (Optional) OpenMP 3 or later

*  (Optional) [Caliper](https://github.com/LLNL/Caliper): a performance profiling/analysis library.


Submodule dependencies:

*  [BLT](https://github.com/LLNL/blt) v0.7.2: a CMake based build system (required)

*  [RAJA](https://github.com/LLNL/RAJA) v2026.07.1: a loop abstraction library (required)

*  [CHAI](https://github.com/LLNL/CHAI) v2026.07.1: a copy hiding abstraction for moving data between memory spaces (optional)

*  [Umpire](https://github.com/LLNL/Umpire) v2026.07.2: a memory management library (required for CUDA/HIP builds and whenever CHAI is enabled)

*  [Caliper](https://github.com/LLNL/Caliper) v2.15.0: performance instrumentation and analysis (used with `ENABLE_CALIPER=ON`).

*  [Adiak](https://github.com/LLNL/Adiak) v0.5.0: application and performance metadata (used with `ENABLE_CALIPER=ON`).

*  [CUB](https://github.com/NVIDIA/cccl): CUDA algorithm primitives required by RAJA for CUDA builds. CUDA 13 places these headers under the toolkit's `include/cccl` directory; Kripke selects that path automatically unless `CUB_DIR` is provided.


Getting Kripke
--------------
Two options are available:
*  Download a released source tarball from github: https://github.com/LLNL/Kripke/releases
*  Clone the source from github.


To check out the v1.2.8 release from Git:

Clone the release tag:
        
        git clone --branch v1.2.8 https://github.com/LLNL/Kripke.git
        
Clone all of the submodules.  The Kripke build system, BLT, resides in 
another repository on github so one must initialize and update the "git submodules"

        cd Kripke
        git submodule update --init --recursive

The released source tarball on github is created with all of the submodules included already.

 

Quick Start
-----------
The easiest way to get Kripke running, is to directly invoke CMake and take whatever system defaults you have for compilers and let CMake find MPI for you.

*  Step 1:  Create a build space (assuming you are starting in the Kripke root directory)
        
        mkdir build

*  Step 2: Run CMake in that build space
        
        cd build
        cmake ..

        For a number of platforms, we have CMake cache files that make things easier:

        cd build
        cmake .. -C ../host-configs/llnl-toss4-MI300A-rocm6-adams.cmake -DCMAKE_BUILD_TYPE=Release

*  Step 3: Now make Kripke:
         
        make -j8
  
*  Step 5: Run Kripke's default problem:
   
        ./bin/kripke.exe
  

Build Options
-------------

Pass these options as `-D<option>=ON` or `-D<option>=OFF` when configuring CMake.
The defaults below apply to a fresh build; host configurations can override them.

| Option | Default | Description |
| --- | --- | --- |
| `ENABLE_MPI` | `OFF` | Enable MPI using CMake's MPI discovery. |
| `ENABLE_OPENMP` | `OFF` | Enable OpenMP CPU threading. |
| `ENABLE_CUDA` | `OFF` | Enable the NVIDIA CUDA backend. |
| `ENABLE_HIP` | `OFF` | Enable the AMD HIP backend. |
| `ENABLE_CHAI` | `OFF` | Use CHAI managed arrays with Umpire. With this option off, GPU builds use explicit Umpire memory management. |
| `ENABLE_GPU_AWARE_MPI` | `OFF` | Pass GPU plane buffers directly to MPI. Requires CUDA or HIP, MPI enabled through `ENABLE_MPI` or `ENABLE_MPI_WRAPPER`, and an MPI installation configured to accept device buffers. |
| `ENABLE_DIRECT_UMPIRE_PLANE_STORAGE` | `OFF` | Allocate I/J/K communication planes directly through Umpire's `NamedAllocationStrategy`, bypassing QuickPool for these fields. Requires `ENABLE_GPU_AWARE_MPI=ON`. |
| `ENABLE_CALIPER` | `OFF` | Enable Caliper instrumentation and Adiak metadata collection. |

For builds using MPI compiler wrappers, `-DENABLE_MPI_WRAPPER=ON` also enables
Kripke's MPI code; configure the corresponding C/C++ compiler wrappers.

GPU-aware MPI works with either CHAI or explicit Umpire memory management.
With GPU-aware MPI disabled, MPI exchanges use host buffers. Direct Umpire plane
storage is optional and provides base allocation pointers for MPI implementations
that need them; the default plane allocator is QuickPool.

For example, enable GPU-aware MPI with explicit Umpire memory management on the
LLNL MI300A system using its host configuration:

```sh
cmake -S . -B build-hip \
  -C host-configs/llnl-toss4-MI300A-rocm6-adams.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CHAI=OFF \
  -DENABLE_GPU_AWARE_MPI=ON
cmake --build build-hip --parallel 8
```

Add `-DENABLE_DIRECT_UMPIRE_PLANE_STORAGE=ON` to that configuration when direct
plane allocations are needed.

When profiling is enabled, the superbuild uses an installed Adiak or builds the
bundled version. Set `adiak_DIR` to an existing Adiak CMake package directory to
select it explicitly. Caliper likewise uses an installed package when found,
otherwise the bundled version is built; `caliper_DIR` selects an existing package.

Host Configurations
-------------------

There are a number of cache init files for LLNL machines and operating systems.  
These might not meet your needs, but can be a very good starting point for developing your own.
A useful subset of the cache init files (located in the ./host-configs/ directory) are:

*  llnl-toss3-clang8.cmake

*  llnl-toss3-intel18.cmake

*  llnl-toss3-gcc8.1.cmake

*  llnl-blueos-P100-nvcc-clang.cmake

*  llnl-blueos-V100-nvcc-clang.cmake

*  llnl-toss4-H100-cuda13-clang-vector.cmake

*  llnl-toss4-H100-cuda13-gcc-vector.cmake

*  llnl-toss4-MI250X-rocm6-vernal.cmake

*  llnl-toss4-MI300A-rocm6-adams.cmake

*  llnl-toss4-intel24.cmake

Running Kripke
==============

Environment Variables
--------------------

If Kripke is built with OpenMP support, then the environment variable ``OMP_NUM_THREADS`` is used to control the number of OpenMP threads.  Kripke does not attempt to modify the OpenMP runtime in any way, so other ``OMP_*`` environment variables should also work as well.

If Kripke is built with Caliper support, Caliper performance measurements can be configured through Caliper environment variables. For example,

    CALI_CONFIG=runtime-report ./kripke ...

will print a time profile of annotated code regions in Kripke. For more information, see https://software.llnl.gov/Caliper.


Command Line Options
--------------------
Command line option help can also be viewed by running `./kripke.exe --help`.

### Problem Size Options:

*   **``--groups <ngroups>``**

    Number of energy groups. (Default: --groups 32)

*   **``--legendre <lorder>``**

    Scattering Legendre Expansion Order (0, 1, ...).  (Default: --legendre 4)

*   **``--quad <ndirs>``**, or **``--quad <polar>:<azim>``**

    Define the quadrature set to use either a fake S2 with <ndirs> points, OR Gauss-Legendre with <polar> by <azim> points.   (Default: --quad 96)

*   **``--zones <x>,<y>,<z>``**

    Number of zones in x,y,z.  (Default: --zones 16,16,16)


### Physics Parameters:

*   **``--sigt <sigt0,sigt1,sigt2>``**
 
    Total material cross-sections.  (Default:   --sigt 0.1,0.0001,0.1)

*   **``--sigs <sigs0,sigs1,sigs2>``**
 
    Scattering material cross-sections.  (Default:   --sigs 0.05,0.00005,0.05)


### On-Node Options:

*   **``--arch <ARCH>``**

    Select the computation backend: Sequential, OpenMP, CUDA, or HIP. Only backends enabled at build time are available. The default is selected at configuration time, with increasing precedence: Sequential, OpenMP, CUDA, HIP.

*   **``--layout <LAYOUT>``**

    Data layout selection.  This determines the data layout and kernel implementation details (such as loop nesting order).  The layouts are determined by the order of unknowns in the angular flux: Direction, Group, and Zone.  Available layouts are DGZ, DZG, GDZ, GZD, ZDG, and ZGD.  The order is specified left-to-right in longest-to-shortest stride.  For example: DGZ means that Directions are the longest stride, and Zones are stride-1.  (Default: --layout DGZ)


*   **``--dev_pool_size <GB>``**

    NOTE: By default, Kripke automatically calculates the Umpire pool size from each rank's problem dimensions, decomposition, material mixtures, solver, and allocation alignment.

    This option allows a user defined Umpire QuickPool size, and overrides the initial automatic pool size per MPI rank calculation. The argument requires a positive integer interpreted in GiB (1 GiB = 2^30 bytes). This option is relevant to CUDA/HIP builds and works with either CHAI or explicit Umpire memory management. It sets the requested initial pool size, not a hard limit on total GPU memory. With direct Umpire plane storage enabled, I/J/K plane allocations are outside this pool.

    For example, the following command requests 8GiB of pool memory for a HIP build of Kripke:

    ```sh
    ./build-hip/kripke.exe --arch HIP --dev_pool_size 8
    ```


### Parallel Decomposition Options:

*   **``--procs <npx,npy,npz>``**
    
    Number of MPI ranks in each spatial dimension. (Default:  --procs 1,1,1)

*   **``--dset <ds>``**

    Number of direction-sets.  Must be a multiple of 8 and divide the number of quadrature points evenly. (Default:  --dset 8)

*   **``--gset <gs>``**
    
    Number of energy group-sets.  Must divide evenly the number energy groups. (Default:  --gset 2)

*   **``--zset <zx>,<zy>,<zz>``**
    
    Number of zone-sets in x, y, and z.  (Default:  --zset 1,1,1)


### Solver Options:

*   **``--niter <NITER>``**

    Number of solver iterations to run. (Default:  --niter 10)

*   **``--pmethod <method>``**

    Parallel solver method. "sweep" for full up-wind sweep (wavefront algorithm). "bj" for Block Jacobi.  (Default: --pmethod sweep)




Future Plans
============

Some ideas for future study:

*   More tuning of CUDA implementation

*   Block AMR

*   More FLOP intensive spatial discretizations such as DFEM's



Links
=====

*  [LLNL Codesign Website](https://codesign.llnl.gov/index.php)


Release
=======

Copyright (c) 2014-26, Lawrence Livermore National Security, LLC.

Produced at the Lawrence Livermore National Laboratory.

All rights reserved.

`LLNL-CODE-775068`  

Unlimited Open Source - BSD Distribution

For release details and restrictions, please read the COPYRIGHT, LICENSE,
and NOTICE files, also linked here:
- [RELEASE](./RELEASE)
- [COPYRIGHT](./COPYRIGHT)
- [LICENSE](./LICENSE)
- [NOTICE](./NOTICE)
