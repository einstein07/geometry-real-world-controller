#!/bin/bash

# The purpose of this script is to create job scripts, for all the experiment runs

# The number of runs
BASE_DIRECTORY="/pfs/work9/workspace/scratch/kn_pop547841-mySpace/update-rate/"
n=100


for ((i=0; i<n; i++)); do
    LAUNCH_FILE="run${i}.sh"
    DIR_i="${BASE_DIRECTORY}run${i}"
    mkdir -p  $DIR_i
    echo "#!/bin/bash" > "${LAUNCH_FILE}"
    echo -e "#SBATCH --job-name=geometry_of_decision_making" >> $LAUNCH_FILE
    echo -e "#SBATCH --nodes=1" >> $LAUNCH_FILE
    echo -e "#SBATCH --ntasks-per-node=1" >> $LAUNCH_FILE
    echo -e "#SBATCH --output=run${i}.out" >> $LAUNCH_FILE
    echo -e "#SBATCH --error=run${i}.err" >> $LAUNCH_FILE
    echo -e "#SBATCH --mem=100G" >> $LAUNCH_FILE
    echo -e "#SBATCH --mail-type=ALL" >> $LAUNCH_FILE
    echo -e "#SBATCH --mail-user=sindiso.mkhatshwa@uni-konstanz.de" >> $LAUNCH_FILE
    echo -e "#SBATCH --time 120:00:00" >> $LAUNCH_FILE
    echo -e "srun apptainer run --bind ${DIR_i}:/mnt /home/kn/kn_kn/kn_pop547841/containers/ros-argos.sif" >> $LAUNCH_FILE
    
    sbatch --partition=single $LAUNCH_FILE
    sleep 120
    
done

