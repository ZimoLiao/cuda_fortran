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
MPIOPT ?=
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC = nvc
RUN = mpiexec
OBJ	= obj
EXE	= exe
MPIOPT = -Mmpi=msmpi
endif
CCFLAGS ?= -fast

all: build run verify

build: mpptest.c gopf.c grph.c ovlap.c pair.c pattern.c util.c getopts.c tstauto.c rate.c mpe_seq.c copy.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c gopf.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c grph.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c ovlap.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c pair.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c pattern.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c util.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c getopts.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c tstauto.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c rate.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c mpe_seq.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c copy.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o mpptest.$(EXE) mpptest.c gopf.$(OBJ) grph.$(OBJ) ovlap.$(OBJ) pair.$(OBJ) pattern.$(OBJ) util.$(OBJ) getopts.$(OBJ) tstauto.$(OBJ) rate.$(OBJ) mpe_seq.$(OBJ) copy.$(OBJ)

run: mpptest.$(EXE)
	$(RUN) -np $(NPROCS) ./mpptest.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
