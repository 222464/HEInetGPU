HEInetGPU
=======

HEInet is similar to HTM (hierarchical temporal memory) and an extension of EInet (Paul D. King et al, [Website](http://redwood.berkeley.edu/w/images/2/29/King_Zylberberg_DeWeese_E_I_Net_Model_of_V1_JNeurosci_2013.pdf)).

It extends EInet by adding hierarchy support, a prediction system, and sparse connectivity.

Overview
-----------

HEInet stands for hierarchical excitatory-inhibitory network. It uses layers consisting of populations of excitatory and inhibitory neurons to perform predictive sparse coding in a hierarchy.

HEInetGPU runs on the GPU, using OpenCL.

Install
-----------

HEInet uses the CMake build system, OpenCL, and SFML for visualization. SFML is not necessary if you are not running the demos.

You can find SFML here: [http://www.sfml-dev.org/download.php](http://www.sfml-dev.org/download.php)

For OpenCL, you will need at least version 1.2 from your hardware vendor.

Usage
-----------

Coming soon! For now, see the demos.

License
-----------

HEInetGPU
Copyright (C) 2015 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgment in the product documentation would be
	appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

------------------------------------------------------------------------------

HEInetGPU uses the following external libraries:

OpenCL
SFML

