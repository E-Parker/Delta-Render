# Delta-Render
A 3D rendering engine written in 100% Python 3.7.x .

I wrote this program to see where i could go with writing a 3D software rendering method.
I will continue to work on this in private, however here's a mostly functional version for anyone
intersted to tinker with. 

Currently, the slowest parts of this code are as follows:

func.Collider()
- Painfully slow even with multithreading, the method i used is just the worst.

render.render() 
- 1 / inv_z calculation done for every pixel drawn to screen.
- screen writes are done one at a time, (array access times are slow)
- overly complex.

make.combineMesh()
- Needed to allow for camera clipping to not perminatly affect the original file.
- uses way to many for loops.

