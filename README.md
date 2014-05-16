CalOptrics
==========

CalOptrics is an open source fast and easy-to-use C++ library for CUDA-based GPU computing for Compuational Optical Imaging purposes. It uses an array-based function set which makes programming with CUDA easier than programming raw CUDA code, so it's main goal is to save time during gpu code development. The idea for CalOptrics was inspired by the GPU computing library ArrayFire: http://arrayfire.com/. Underneath CalOptrics is powered by the Open Source libraries Thrust, Cusp, and other cuda open source libraries. The goal was for this codebase to be designed in an intelligent manner (using Object-oriented and abstraction techniques like interfaces to make modifications to raw lower-level implementations simple and quick and requiring no change to the code above it that uses it since that code will use the interfaces). The design also makes it easy to implement a non-CUDA version like OpenCL to be used in this library.

I hope that a clean design strategy like this will encourage others to join, contribute, and improve upon any of the present implementations in the project so that if I leave the project, others will carry it on. 

If you have any questions or concerns please email me at either diivanand@berkeley.edu or diivanand@gmail.com.
