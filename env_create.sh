mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' > \
    $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$OLD_LD_LIBRARY_PATH' >> \
    $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' > \
    $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> \
    $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
