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
RUN = mpirun
OBJ = o
EXE = out
MPIOPT  =
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC = nvc
MPIOPT ?= -Mmpi=msmpi
RUN = mpiexec
OBJ = obj
EXE = exe
endif

CCFLAGS ?= -fast

all: build run verify

build: goptest.c getopts.c gopf.c grph.c rate.c util.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c gopf.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c grph.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c rate.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c util.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c getopts.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o gop.$(EXE) goptest.c getopts.$(OBJ) gopf.$(OBJ) grph.$(OBJ) rate.$(OBJ) util.$(OBJ)

run: gop.$(EXE)
	$(RUN) -np $(NPROCS) ./gop.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
