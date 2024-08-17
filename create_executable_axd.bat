rmdir /s /q dist
rmdir /s /q build

pyinstaller aEDXD.spec

cd dist/axd*

rm mkl_avx512.dll, ^
    mkl_avx2.dll, ^
    mkl_avx.dll, ^
    mkl_mc3.dll, ^
    mkl_mc.dll, ^
    mkl_pgi_thread.dll, ^
    mkl_tbb_thread.dll, ^
    mkl_sequential.dll, ^
    mkl_vml_mc.dll, ^
    mkl_vml_mc2.dll, ^
    mkl_vml_mc3.dll, ^
    mkl_vml_avx.dll, ^
    mkl_vml_avx2.dll, ^
    mkl_vml_avx512.dll, ^
    mkl_scalapack_ilp64.dll, ^
    mkl_scalapack_lp64.dll, ^
    Qt5Quick.dll, ^
    Qt5Qml.dll

rmdir /s /q bokeh, ^
    bottleneck, ^
    certify, ^
    cryptography, ^
    Cython, ^
    cytoolz, ^
    etc, ^
    docutils, ^
    gevent, ^
    jedi, ^
    lxml, ^
    nbconvert, ^
    nbformat, ^
    notebook, ^
    mpl-data, ^
    matplotlib, ^
    mkl-fft, ^
    msgpack, ^
    pandas, ^
    share, ^
    PyQt6\Qt\bin, ^
    PyQt6\Qt\uic


aEDXD_run.exe
cd ..
cd ..