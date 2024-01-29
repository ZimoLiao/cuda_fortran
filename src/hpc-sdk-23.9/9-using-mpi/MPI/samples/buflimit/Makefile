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
RUN = mpiexec
OBJ = obj
EXE = exe
MPIOPT ?= -Mmpi=msmpi
endif

CCFLAGS ?= -fast -DHAVE_STDLIB_H

all: build run verify

build: buflimit.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o buflimit.$(EXE) buflimit.c

run: buflimit.$(EXE)
	$(RUN) -np 2 ./buflimit.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
