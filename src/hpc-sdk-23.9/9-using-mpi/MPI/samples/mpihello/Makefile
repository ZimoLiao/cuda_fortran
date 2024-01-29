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
MPICC = mpicc
MPIFC = mpif90
RUN = mpirun
OBJ = o
EXE = out
MSLIB =
Bdynamic = -Bdynamic
MPIOPT ?=
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC = nvc
MPIFC = nvfortran
MPIOPT = -Mmpi=msmpi
RUN = mpiexec
OBJ = obj
EXE	= exe
MSLIB = Ws2_32.lib
Bdynamic =
endif
ifeq ($(findstring Darwin, $(UNAME)), Darwin)
Bdynamic =
endif

FCFLAGS ?= -fast
CCFLAGS ?= -fast

all: build run verify

build: mpihello.f myname.c
	$(MPIFC) $(MPIOPT) $(FCFLAGS) -o mpihello.$(EXE) mpihello.f
	$(MPIFC) $(MPIOPT) $(FCFLAGS) -o mpihello_f90.$(EXE) mpihello.f90
	$(MPICC) $(MPIOPT) $(CCFLAGS) $(Bdynamic) -o myname.$(EXE) myname.c $(MSLIB)

run: mpihello.$(EXE) mpihello_f90.$(EXE) myname.$(EXE)
	@echo "--------------- Executing mpihello.$(EXE) ----------------------"
	$(RUN) -np $(NPROCS) ./mpihello.$(EXE)
	@echo "--------------- Executing mpihello_f90.$(EXE) ------------------"
	$(RUN) -np $(NPROCS) ./mpihello_f90.$(EXE)
	@echo "--------------- Executing myname.$(EXE) ------------------------"
	$(RUN) -np $(NPROCS) ./myname.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
