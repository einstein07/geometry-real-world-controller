#!/bin/bash

# The purpose of this script is to create job scripts for all experiment runs.

BASE_DIRECTORY="/pfs/work9/workspace/scratch/kn_pop547841-mySpace/update-rate/"
n=1
IMAGE="/home/kn/kn_kn/kn_pop547841/containers/tb4.sif"

for ((i=0; i<n; i++)); do
    LAUNCH_FILE="run${i}.sh"
    DIR_i="${BASE_DIRECTORY}run${i}"
    mkdir -p "$DIR_i"

    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=geometry_of_decision_making"
        echo "#SBATCH --partition=cpu"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --ntasks-per-node=1"
        echo "#SBATCH --output=run${i}.out"
        echo "#SBATCH --error=run${i}.err"
        echo "#SBATCH --mem=100G"
        echo "#SBATCH --mail-type=ALL"
        echo "#SBATCH --mail-user=sindiso.mkhatshwa@uni-konstanz.de"
        echo "#SBATCH --time=72:00:00"
        echo ""
        echo "set -euo pipefail"
        echo "export TMPDIR=/tmp"
        echo "mkdir -p \"\$TMPDIR\""
        echo "srun --export=ALL,TMPDIR=/tmp apptainer run --bind ${DIR_i}:/mnt ${IMAGE}"
    } > "$LAUNCH_FILE"

    sbatch "$LAUNCH_FILE"
    sleep 120
done
