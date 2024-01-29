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
BUFSIZE = 100000000
UNAME := $(shell uname -a)
MPICC = mpicc
RUN = mpirun
OBJ = o
EXE = out
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
      MPICC = nvc
      RUN = mpiexec
      OBJ = obj
      EXE = exe
      CCFLAGS += -stack:10000000
      MPIOPT  = -Mmpi=msmpi
endif

MPIOPT ?=
CCFLAGS ?= -fast -DHAVE_STDLIB_H

all: build run verify

build: copytest.c copytest2.c duff.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o copytest.$(EXE) copytest.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o copytest2.$(EXE) copytest2.c

run: copytest.$(EXE) copytest2.$(EXE)
	@echo "------------------- Running copytest.$(EXE) ---------------------------"
	$(RUN) -np $(NPROCS) ./copytest.$(EXE) $(BUFSIZE)
	@echo "------------------- Running copytest2.$(EXE) ---------------------------"
	$(RUN) -np $(NPROCS) ./copytest2.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
