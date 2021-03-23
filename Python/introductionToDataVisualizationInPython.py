'''Plotting multiple graphs 
Strategies
-plotting many graphs on common axes
-creating axes within a figure
-creating subplots within a figure 

import matplotlib.pyplot as plt

plt.plot(t, temperature, 'r') #r for red
#Appears on the same axes
plt.plot(t, dewpoint, 'b') #b for blue
plt.xlabel('Date')
plt.title('Temperature & Dew Point')
#Renders plot objects to screen 
plt.show()

The tool to construct axes explicitly is the axes() command. 

plt.axes([0.05, 0.05, 0.425, 0.9]) #construct the left side of the axes 
plt.plot(t, temperature, 'r')
plt.xlabel('Date')
plt.title('Temperature')
plt.axes([0.525,0.05,0.425,0.9]) #makes new axes on the right of the figure 
plt.plot(t, dewpoint, 'b')
plt.xlabel('Date')
plt.title('Dew Point')
plt.show()

-this displays two graphs side-by-side 

The axes() command
-Syntax axes(): the axes command requires the lower left corner 
suntax: axes( [x_lo, y_lo, width, height] ) # requires the coordinate of the left lower corner 
Units between 0 and 1 (figure dimensions)

The subplot command creates a grid of axes, freeing us from figuring out 

plt.subplot(2, 1, 1)
plt.plot(t, temperature, 'r')
plt.xlabel('Date')
plt.title('Temperature')

plt.subplot(2, 1, 2)
plt.plot(t, dewpoint, 'b')
plt.xlabel('Date')
plt.title('Dew Point')

plt.tight_layout()
plt.show()

The subplot() command
Syntax: subplot(nrows, ncols, nsubplot)
Subplot ordering:
-Row-wise from top left
-Indexed from 1
'''
###Multiple plots 
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()

###Using axes()
# Create plot axes for the first line plot
plt.axes([0.05, 0.05, 0.425, 0.9])

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Create plot axes for the second line plot
plt.axes([0.525, 0.05, 0.425, 0.9])

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()


###Using subplot()(1)
# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1, 2, 1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1, 2, 2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()

##Using subplot()(2)
# Create a figure with 2x2 subplot layout and make the top left subplot active
plt.subplot(2, 2, 1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the top right subplot active in the current 2x2 subplot grid 
plt.subplot(2, 2, 2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Make the bottom left subplot active in the current 2x2 subplot grid
plt.subplot(2, 2, 3)

# Plot in green the % of degrees awarded to women in Health Professions
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Make the bottom right subplot active in the current 2x2 subplot grid
plt.subplot(2, 2, 4)

# Plot in yellow the % of degrees awarded to women in Education
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()


'''
Zooming in on a specific region of a graph can be achieved with axis(), xlim(), ylim()

Controlling axis extents
axis([xmin, xmax, ymin, ymax])
Control over individual axis extents

xlim([xmin, xmax])
ylim([ymin, ymax])

Can use tuples, list for extents 
e.g. xlim((-2, 3)) works
e.g. xlim([-2, 3]) works also

GDP over time
import matplotlib.pyplot as plt
plt.plot(yr, gdp)
plt.xlabel('Year')
plt.xlabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.show()

Zooming in to a psecific region of the graph generated 
Using xlim()
plt.plot(yr, gdp)
plt.xlabel('Year')
plt.ylabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.xlim((1947, 1957))

plt.show()

Using xlim() and ylim()
plt.plot(yr, gdp)
plt.xlabel('Year')
plt.xlabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.xlim((1947, 1957))
plt.ylim((0, 1000))

plt.show()

Now we can set the horizonatal limits and the vertical limits
plt.plot(yr, gdp)
plt.xlable('Year')
plt.ylabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.axis((1947, 1957, 0, 600))

plt.show()


Other axis() options

Invocation          Result
axis('off')         turns off axis lines, labels
axis('equal')       equal scaling on x and y axes
axis('square')      forces square plot
axis('tight')       sets xlim, ylim to show all data

Using axis('equal')
plt.sublot(2, 1, 1)
plt.plot(x, y, 'red')
plt.title('default axis')
plt.subplot(2, 1, 2)
plt.plot(x, y, 'red')

plt.axis('equal')

plt.title('axis equal')
plt.tight_layout()
plt.show()
    

'''

###Using xlim(), ylim()
# Plot the % of degrees awarded to women in Computer Science and the Physical Sciences
plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')

# Add the axis labels
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')

# Set the x-axis range
plt.xlim((1990, 2010))

# Set the y-axis range
plt.ylim((0, 50))

# Add a title and display the plot
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()

# Save the image as 'xlim_and_ylim.png'
plt.savefig('xlim_and_ylim.png')

###Using axis()
# Plot in blue the % of degrees awarded to women in Computer Science
plt.plot(year,computer_science, color='blue')

# Plot in red the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color='red')

# Set the x-axis and y-axis limits
plt.axis((1990, 2010, 0, 50))

# Show the figure
plt.show()

# Save the figure as 'axis_limits.png'
plt.savefig('axis_limits.png')


'''
Legends, annotations and styles 

Legends - provide labels for overlaid points and curves 


import matplotlib.pypot as plt 
plt.scatter(setosa_len, setosa_wid, 
            marker='o', color='red', label='setosa')

plt.scatter(setosa_len, setosa_wid, 
            marker='o', color='green', label='versicolor')

plt.scatter(setosa_len, setosa_wid, 
            marker='o', color='blue', label='virginica')

plt.legend(loc='upper right')
plt.title('Iris data')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

Legend locations 
string          code 
'upper left'    2
'center left'   6
'lower left'    3
'upper center'  9
'center'        10
'lower center'  8

string          code
'upper right'   1
'center right'  7
'lower right'   4
'right'         5
'best'          0

The annotate function adds text to a figure 
Text labels and arrows using annotate() method
There are flexible ways to specify coordinate 
keyword arrowprops:dict of arrow properties 
    - width
    - color
    - etc.

Using annotate() for text
plt.annotate('setosa', xy=(5.0, 3.5))
plt.annotate('virginica', xy=(7.25, 3.5))
plt.annotate('versicolor', xy=(5.0, 2.0))
plt.show()

Options for annotate()
options     description
s           text of label
xy          coordinates to annotate
xytext      coordinates of label
arrowprops  cotrols drawing of arrow


plt.annotate('setosa', xy=(5.0, 3.5),
                xytext=(4.25, 4.0),
                arrowprops={'color':'red'})
plt.annotate('virginica', xy=(7.2, 3.6),
                xytext=(6.5, 4.0),
                arrowprops={'color':'blue})
plt.annotate('versicolor', xy=(5.05, 1.95),
                xytext=(5.5, 1.75),
                arrowprops={'color':'green'})
plt.show()


Working with plot styles 
style sheets in matplotlib
Defaults for lines, points, backgrounds, etc.
Switch styles globally with plt.style.use()
plt.style.available:list of styles

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.style.use('fivethirtyeight')

'''

###Using legend()
# Specify the label 'Computer Science'
plt.plot(year, computer_science, color='red', label='Computer Science') 

# Specify the label 'Physical Sciences' 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')

# Add a legend at the lower center
plt.legend(loc='lower center')

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()


###Using annotate()
# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()

# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.argmax()]

# Plot with legend as before
plt.plot(year, computer_science, color='red', label='Computer Science') 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

# Add a black arrow annotation
plt.annotate('Maximum', xy=((yr_max, cs_max)), xytext=((yr_max+5, cs_max+5)), arrowprops=dict(facecolor='k'))

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()

###
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Set the style to 'ggplot'
plt.style.use('ggplot')

# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1) 

# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()

'''
Working with 2D arrays aka raster data - represent either images or 
functions of two variables, also known as bivariate funtions 

Numpy arrays
-Homogeneous in type
-Calculations all at once
They support vectorized computations or calculations over the entire array without 
writing loops 
-Indexing with brackets:
**A[index] for 1D array
**A[index0, index1] for 2D array
-Slicing: 1D arrays: A[slice], 2D arrays: A[slice0, slice1]
**splcing; slice = start:stop:stride
indexes from start to stop-1 in steps of stride

Using meshgrid()
meshgrids.py:

impoort numpy as np
u = p.linspace(-2,2,3)
v = np.linspace(-1,1,5)
X,Y = np.meshgrid(u, v)

import numpy as np
import matplotlib.pyplot as plt
u = np.linspace(-2, 2, 3)
v = np.linspace(-1, 1, 5)
X, Y = np.meshgrid(u, v)

Z = X**2/25 + Y**2/4

print(Z)
plt.set_cmap('gray')
plt.pcolor(Z)
plt.show()
#dark pixels on the graph are closer to 0 than the lighter squares 

Orientations of 2D arrays & images 
orientation.py

import numpy as np
import matplotlib.pyplot as plt
Z = np.array([[1, 2, 3], [4, 5, 6]])
print(z)
plt.pcolor(Z)
plt.show()

When pcolor() plots pixels, values increase from 1 to 6 with values increasing from 
left to right, then vertically from bottom to top starting from the bottom left corner 


'''

###Generating meshes 
# Import numpy and matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

# Generate two 1-D arrays: u, v
u = np.linspace(-2, 2, 41)
v = np.linspace(-1, 1, 21)

# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u, v)

# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 

# Display the resulting image with pcolor()
plt.pcolor(Z)
plt.show()

# Save the figure to 'sine_mesh.png'
plt.savefig('sine_mesh.png')


plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()

-produces a psuedocolor plot using array A. 