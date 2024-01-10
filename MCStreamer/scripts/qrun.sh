#! /bin/bash
# Script to start a simulation in the qsub/slurm queue
# Alejandro Luque - 2012

JULIA_EXEC=julia
UNAME_FULL=`uname -a`
DATE=`date`

echo "# [${DATE}]"
echo "# ${UNAME_FULL}"

echo "Executing" ${JULIA_EXEC} --check-bounds=no --startup-file=no --project=${PROJECT_PATH} ${MAIN} ${INPUT_FILE}

export JULIA_NUM_THREADS=${NTHREADS}
${JULIA_EXEC} --check-bounds=no --startup-file=no --project=${PROJECT_PATH} ${MAIN} ${INPUT_FILE}

DATE=`date`
echo "# [${DATE}]"

