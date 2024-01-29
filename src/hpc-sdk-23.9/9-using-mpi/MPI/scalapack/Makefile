# SPDX-FileCopyrightText: Copyright (c) 2017 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

NPROCS = 2
UNAME := $(shell uname -a)
MPIFC = mpif90
RUN = mpirun
OBJ = o
EXE = out
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC = nvc -Mmpi=msmpi
MPIFC = nvfortran -Mmpi=msmpi
RUN = mpiexec
OBJ = obj
EXE	= out
endif

FCFLAGS ?= -fast

all: build run verify

build: pdgetrrv.f pdlafchk.f pdlaschk.f pdludriver.f pdluinfo.f pdmatgen.f pmatgeninc.f
	$(MPIFC) $(FCFLAGS) -c pdgetrrv.f
	$(MPIFC) $(FCFLAGS) -c pdlafchk.f
	$(MPIFC) $(FCFLAGS) -c pdlaschk.f
	$(MPIFC) $(FCFLAGS) -c pdluinfo.f
	$(MPIFC) $(FCFLAGS) -c pdmatgen.f
	$(MPIFC) $(FCFLAGS) -c pmatgeninc.f
	$(MPIFC) $(FCFLAGS) -o xdlu.$(EXE) pdludriver.f pdgetrrv.$(OBJ) pdlafchk.$(OBJ) pdlaschk.$(OBJ) pdluinfo.$(OBJ) pdmatgen.$(OBJ) pmatgeninc.$(OBJ) -Mscalapack

run: xdlu.$(EXE)
	$(RUN) -np $(NPROCS) ./xdlu.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
