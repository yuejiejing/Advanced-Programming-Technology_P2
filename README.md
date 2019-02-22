# Advanced-Programming-Technology_P2
2D and 3D Heat Diffusion Simulation using CUDA

You need to implement a CUDA program that will output the temperature at
each grid point after running the equation for a specified number of timesteps.

Example execution
./heat2D3D sample.conf

sample.conf is a configuration file which will specify the parameters for your simulation. Below are example con-
guration files for 2 and 3 dimensions.

##### 2D CONFIGURATION EXAMPLE #####
#your code should ignore lines starting with ’#’
#you can assume arguments will always follow this ordering however whitespace may vary
#2D or 3D
2D
#the value for k
4
#number of timestep to run
200
#width (x-axis) and height (y-axis) of grid
800,800
#default starting temperature for nodes
5
#list of fixed temperature blocks (squares for 2D)
#can be 0, 1 or more
#assume blocks won’t overlap
#location_x, location_y, width, height, fixed temperature
5, 5, 20, 20, 200
500, 500, 10, 10, 300

##### 3D CONFIGURATION EXAMPLE #####
#your code should ignore lines starting with ’#’
#you can assume arguments will always follow this ordering however whitespace may vary
#2D or 3D
3D
#the value for k
4
#number of timestep to run
200
#width (x-axis) height (y-axis), depth (z-axis) of grid
800,800,800
#default starting temperature for nodes
5
#list of fixed temperature blocks (cubes for 3D)
#can be 0, 1 or more
#assume cubes won’t overlap
#location_x, location_y, location_z, width, height, depth, fixed temperature
5, 5, 5, 20, 20, 20, 200
500, 500, 500, 10, 10, 10, 300

You may design your program however you wish as long as it is reasonably efficient. We will discuss possible
implementations in class. At the end of execution, your program should write an output file named heatOutput.csv
listing the temperatures at each grid point in comma separated format. Use floats for all values.
