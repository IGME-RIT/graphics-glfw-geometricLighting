/*
Geometric Lighting 
(c) 2015
original authors: Benjamin Robbins
Written under the supervision of David I. Schwartz, Ph.D., and
supported by a professional development seed grant from the B. Thomas
Golisano College of Computing & Information Sciences
(https://www.rit.edu/gccis) at the Rochester Institute of Technology.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*	This example makes use of the teapot bezier spline established in the B-Spline example and thus retains many of the same components.
*	The two main exceptions are the inclusion of normals in the teapot verticies and the use of some basic lighting. The normals for each
*	of the vertices in the Bezier patches are computed using factors from the secondo order Bernstein polynomial: 
*	(1 - t)^2*A + 2t(1 - t)*B + t^2*C
*	Where t is the location on the curve and A, B, and C are the differences between the four control points on the Bezier curve. Using
*	this you can solve for the tangent to the bezier curve. By computing tangents for both rows and columns of the patch, you can use
*	a cross product to find the normal to the surface. 
*	These normals are then used in the fragment shader to find the light intensity for each pixel based on the normal at that point and
*	the direction of incoming light. 
*	There are 3 static component classes that make up the base functionality of this program. 
*	
*	1) RenderManager
*	- This class maintains the display list for the scene being rendered and thus handles the processes of updating and drawing all 
*	of the RenderShapes that have been instantiated in the scene.
*
*	2) CameraManager
*	- This class maintains data relating to the view and projection matrices used in the rendering pipeline. It also handles updating
*	this data based on user input.
*
*	3) InputManager
*	- This class maintains data for the current state of user input for the mouse and keyboard.
*
*	B_Spline
*	- This non-static class is instantiated to maintain an array of Patch objects. Control point data is sent to this class to manipulate
*	component patches.
*
*	Patch
*	- This non-static class handles the data storage and updating for a single bezier surface containing a dynamically drawn RenderShape
*	and 16 control points that through a third-order Bernstein polynomial determine mathematically the positions of the vertices comprising
*	the surface. 
*
*	RenderShape 
*	- This class tracks instance data for every shape that is drawn to the screen. This data primarily includes a vertex array object and
*	transform data. This transform data is used to generate the model matrix used along with the view and projection matrices in the 
*	rendering pipeline.
*	
*	Init_Shader
*	- Contains static functions for reading, compiling and linking shaders.
*
*
*	SHADERS
*
*	vShader.glsl
*	- Simple through shader, applies transforms to verts and normals before passing them through to the fragment shader.
*
*	fShader.glsl
*	- Uses a hard-coded point-light to apply the color of the light to the current fragment based on lambert's law of cosines.
*	see: http://en.wikipedia.org/wiki/Lambert's_cosine_law
*	Basically, the brightness of a surface is determined by the dot product of the normal of the surface and the direction of the 
*	light. This dot product is equal to the cosine of the angle between the two vectors. Since it is a cosine, it will be a value between
*	-1 and 1. Clamp this value to between 0 and 1 and you have a diffuse brightness value. Multiply this by the color and power of the light 
*	and you have a lit surface. However the intensity of diffuse light drops of significantly the further from the light source you get.
*	Since the light expands in a sphere and the area of a sphere equals 4 * pi * radius^2, we can simply say that the intensity of the light
*	at a given distance from the source equals the base intensity of the light divided by the square of the distance from the light.
*	So Ldiffuse = clamp(Normal dot L, 0, 1) * diffuseColor * diffusePower / distance^2;
*/

#include <GLEW\GL\glew.h>
#include <GLFW\glfw3.h>
#include <GLM\gtc\type_ptr.hpp>
#include <GLM\gtc\matrix_transform.hpp>
#include <GLM\gtc\quaternion.hpp>
#include <GLM\gtc\random.hpp>
#include <iostream>
#include <ctime>

#include "RenderShape.h"
#include "Init_Shader.h"
#include "RenderManager.h"
#include "InputManager.h"
#include "B-Spline.h"
#include "Patch.h"
#include "CameraManager.h"

GLFWwindow* window;

// Shader buffer pointers
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;

// Shader uniform locations
GLint uMPMat;
GLint uMPVMat;
GLint uColor;


// Source http://www.holmes3d.net/graphics/teapot/teapotCGA.bpt
GLfloat teapotControlPoints[] = {
#pragma region Teapot Control Points
	1.4, 2.25, 0,		1.3375,	2.38125, 0,			1.4375,	2.38125, 0,			1.5, 2.25, 0,
	1.4, 2.25, .784,	1.3375,	2.38125, .749,		1.4375, 2.38125, .805,		1.5, 2.25, .84,
	.784, 2.25, 1.4,	.749, 2.38125, 1.3375,		.805, 2.38125, 1.4375,		.84, 2.25, 1.5,
	0, 2.25, 1.4,		0, 2.38125, 1.3375,			0, 2.38125, 1.4375,			0, 2.25, 1.5,

	0, 2.25, 1.4,		0, 2.38125, 1.3375,			0, 2.38125, 1.4375,			0, 2.25, 1.5,
	-.784, 2.25, 1.4,	-.749, 2.38125, 1.3375,		-.805, 2.38125,	1.4375,		-.84, 2.25, 1.5,
	-1.4, 2.25, .784,	-1.3375, 2.38125, .749,		-1.4375, 2.38125, .805,		-1.5, 2.25, .84,
	-1.4, 2.25, 0,		-1.3375, 2.38125, 0,		-1.4375, 2.38125, 0,		-1.5, 2.25, 0,

	-1.4, 2.25, 0,		-1.3375, 2.38125, 0,		-1.4375, 2.38125, 0,		-1.5, 2.25, 0,
	-1.4, 2.25, -.784,	-1.3375, 2.38125, -.749,	-1.4375, 2.38125, -.805,	-1.5, 2.25, -.84,
	-.784, 2.25, -1.4,	-.749, 2.38125, -1.3375,	-.805, 2.38125,	-1.4375,	-.84, 2.25, -1.5,
	0, 2.25, -1.4,		0, 2.38125, -1.3375,		0,	2.38125, -1.4375,		0, 2.25, -1.5,

	0, 2.25, -1.4,		0, 2.38125, -1.3375,		0, 2.38125, -1.4375,		0, 2.25, -1.5,
	.784, 2.25, -1.4,	.749, 2.38125, -1.3375,		.805, 2.38125, -1.4375,		.84, 2.25, -1.5,
	1.4, 2.25, -.784,	1.3375,	2.38125, -.749,		1.4375, 2.38125, -.805,		1.5, 2.25, -.84,
	1.4, 2.25, 0,		1.3375,	2.38125, 0,			1.4375, 2.38125, 0,			1.5, 2.25, 0,

	1.5, 2.25, 0,		1.75, 1.725, 0,				2, 1.2,	0,					2, .75, 0,
	1.5, 2.25, .84,		1.75, 1.725, .98,			2, 1.2,	1.12,				2, .75, 1.12,
	.84, 2.25, 1.5,		.98, 1.725, 1.75,			1.12, 1.2, 2,				1.12, .75, 2,
	0, 2.25, 1.5,		0, 1.725, 1.75,				0, 1.2,	2,					0, .75, 2,

	0, 2.25, 1.5,		0, 1.725, 1.75,				0, 1.2,	2,					0, .75, 2,
	-.84, 2.25, 1.5,	-.98, 1.725, 1.75,			-1.12, 1.2, 2,				-1.12, .75, 2,
	-1.5, 2.25, .84,	-1.75, 1.725, .98,			-2, 1.2, 1.12,				-2, .75, 1.12,
	-1.5, 2.25, 0,		-1.75, 1.725, 0,			-2, 1.2, 0,					-2, .75, 0,

	-1.5, 2.25, 0,		-1.75, 1.725, 0,			-2, 1.2, 0,					-2, .75, 0,
	-1.5, 2.25, -.84,	-1.75, 1.725, -.98,			-2, 1.2, -1.12,				-2, .75, -1.12,
	-.84, 2.25, -1.5,	-.98, 1.725, -1.75,			-1.12, 1.2, -2,				-1.12, .75, -2,
	0, 2.25, -1.5,		0, 1.725,-1.75,				0, 1.2, -2,					0, .75, -2,

	0, 2.25, -1.5,		0, 1.725, -1.75,			0, 1.2,	-2,					0, .75, -2,
	.84, 2.25, -1.5,	.98, 1.725, -1.75,			1.12, 1.2, -2,				1.12, .75, -2,
	1.5, 2.25, -.84,	1.75, 1.725, -.98,			2, 1.2, -1.12,				2, .75, -1.12,
	1.5, 2.25, 0,		1.75, 1.725, 0,				2, 1.2, 0,					2, .75, 0,

	2, .75, 0,			2, .3, 0,					1.5, .075, 0,				1.5, 0, 0,
	2, .75, 1.12,		2, 	.3, 1.12,				1.5, .075, .84,				1.5, 0, .84,
	1.12, .75, 2,		1.12, .3, 2,				.84, .075, 1.5,				.84, 0, 1.5,
	0, .75, 2,			0, .3, 2,					0, .075, 1.5,				0, 0, 1.5,

	0, .75, 2,			0, .3, 2,					0, .075, 1.5,				0, 0, 1.5,
	-1.12, .75, 2,		-1.12, .3, 2,				-.84, .075, 1.5,			-.84, 0, 1.5,
	-2, .75, 1.12,		-2, .3, 1.12,				-1.5, .075,	.84,			-1.5, 0, .84,
	-2, .75, 0,			-2, .3, 0,					-1.5, .075,	0,				-1.5, 0, 0,

	-2, .75, 0,			-2,	.3, 0,					-1.5, .075,	0,				-1.5, 0, 0,
	-2, .75, -1.12,		-2,	.3, -1.12,				-1.5, .075,	-.84,			-1.5, 0, -.84,
	-1.12, .75, -2,		-1.12, .3, -2,				-.84, .075,	-1.5,			-.84, 0, -1.5,
	0, .75, -2,			0, .3, -2,					0, .075, -1.5,				0, 0, -1.5,

	0, .75, -2,			0, .3, -2,					0, .075, -1.5,				0, 0, -1.5,
	1.12, .75, -2,		1.12, .3, -2,				.84, .075, -1.5,			.84, 0, -1.5,
	2, .75, -1.12,		2, .3, -1.12,				1.5, .075,	-.84,			1.5, 0, -.84,
	2, .75, 0,			2, .3, 0,					1.5, .075,	0,				1.5, 0, 0,

	-1.6, 1.875, 0,		-2.3, 1.875, 0,				-2.7, 1.875, 0,				-2.7, 1.65, 0,
	-1.6, 1.875, .3,	-2.3, 1.875, .3,			-2.7, 1.875, .3,			-2.7, 1.65, .3,
	-1.5, 2.1, .3,		-2.5, 2.1, .3,				-3, 2.1, .3,				-3, 1.65, .3,
	-1.5, 2.1, 0,		-2.5, 2.1, 0,				-3, 2.1, 0,					-3, 1.65, 0,

	-1.5, 2.1, 0,		-2.5, 2.1, 0,				-3, 2.1, 0,					-3, 1.65, 0,
	-1.5, 2.1, -.3,		-2.5, 2.1, -.3,				-3, 2.1, -.3,				-3, 1.65, -.3,
	-1.6, 1.875, -.3,	-2.3, 1.875, -.3,			-2.7, 1.875, -.3,			-2.7, 1.65, -.3,
	-1.6, 1.875, 0,		-2.3, 1.875, 0,				-2.7, 1.875, 0,				-2.7, 1.65, 0,

	-2.7, 1.65, 0,		-2.7, 1.425, 0,				-2.5, .975, 0,				-2, .75, 0,
	-2.7, 1.65, .3,		-2.7, 1.425, .3,			-2.5, .975,	.3,				-2, .75, .3,
	-3, 1.65, .3,		-3, 1.2, .3,				-2.65, .7875, .3,			-1.9, .45, .3,
	-3, 1.65, 0,		-3,	1.2, 0,					-2.65, .7875, 0,			-1.9, .45, 0,

	-3, 1.65, 0,		-3,	1.2, 0,					-2.65, .7875, 0,			-1.9, .45, 0,
	-3, 1.65, -.3,		-3,	1.2, -.3,				-2.65, .7875, -.3,			-1.9, .45, -.3,
	-2.7, 1.65, -.3,	-2.7, 1.425, -.3,			-2.5, .975, -.3,			-2, .75, -.3,
	-2.7, 1.65, 0,		-2.7, 1.425, 0,				-2.5, .975,	0,				-2, .75, 0,

	1.7, 1.275, 0,		2.6, 1.275, 0,				2.3, 1.95, 0,				2.7, 2.25, 0,
	1.7, 1.275, .66,	2.6, 1.275, .66,			2.3, 1.95,	.25,			2.7, 2.25, .25,
	1.7, .45, .66,		3.1, .675, .66,				2.4, 1.875,	.25,			3.3, 2.25, .25,
	1.7, .45, 0,		3.1, .675, 0,				2.4, 1.875,	0,				3.3, 2.25, 0,

	1.7, .45, 0,		3.1, .675, 0,				2.4, 1.875,	0,				3.3, 2.25, 0,
	1.7, .45, -.66,		3.1, .675, -.66,			2.4, 1.875,	-.25,			3.3, 2.25, -.25,
	1.7, 1.275, -.66,	2.6, 1.275, -.66,			2.3, 1.95, -.25,			2.7, 2.25, -.25,
	1.7, 1.275, 0,		2.6, 1.275, 0,				2.3, 1.95, 0,				2.7, 2.25, 0,

	2.7, 2.25, 0,		2.8, 2.325, 0,				2.9, 2.325, 0,				 2.8, 2.25, 0,
	2.7, 2.25, .25,		2.8, 2.325, .25,			2.9, 2.325,	.15,			2.8, 2.25, .15,
	3.3, 2.25, .25,		3.525, 2.34375, .25,		3.45, 2.3625, .15,			3.2, 2.25, .15,
	3.3, 2.25, 0,		3.525, 2.34375, 0,			3.45, 2.3625, 0,			3.2, 2.25, 0,

	3.3, 2.25, 0,		3.525, 2.34375, 0,			3.45, 2.3625, 0,			3.2, 2.25, 0,
	3.3, 2.25, -.25,	3.525, 2.34375, -.25,		3.45, 2.3625, -.15,			3.2, 2.25, -.15,
	2.7, 2.25, -.25,	2.8, 2.325, -.25,			2.9, 2.325, -.15,			2.8, 2.25, -.15,
	2.7, 2.25, 0,		2.8, 2.325, 0,				2.9, 2.325,	0,				2.8, 2.25, 0,

	0, 3, 0,			.8, 3, 0,					0, 2.7, 0,					.2, 2.55, 0,
	0, 3, .002,			.8,	3, .45,					0, 2.7,	0,					.2, 2.55, .112,
	.002, 3, 0,			.45, 3, .8,					0, 2.7,	0,					.112, 2.55, .2,
	0, 3, 0,			0, 3, .8,					0, 2.7,	0,					0, 2.55, .2,

	0, 3, 0,			0, 3, .8,					0, 2.7,	0,					0, 2.55, .2,
	-.002, 3, 0,		-.45, 3, .8,				0, 2.7,	0,					-.112, 2.55, .2,
	0, 3, .002,			-.8, 3, .45,				0, 2.7,	0,					-.2, 2.55, .112,
	0, 3, 0,			-.8, 3, 0,					0, 2.7,	0,					-.2, 2.55, 0,

	0, 3, 0,			-.8, 3, 0,					0, 2.7,	0,					-.2, 2.55, 0,
	0, 3, -.002,		-.8, 3, -.45,				0, 2.7,	0,					-.2, 2.55, -.112,
	-.002, 3, 0,		-.45, 3, -.8,				0, 2.7,	0,					-.112, 2.55, -.2,
	0, 3, 0,			0, 3, -.8,					0, 2.7,	0,					0, 2.55, -.2,

	0, 3, 0,			0, 3, -.8,					0, 2.7,	0,					0, 2.55, -.2,
	.002, 3, 0,			.45, 3, -.8,				0, 2.7,	0,					.112, 2.55, -.2,
	0, 3, -.002,		.8,	3, -.45,				0, 2.7, 0,					.2, 2.55, -.112,
	0, 3, 0,			.8, 3, 0,					0, 2.7,	0,					.2, 2.55, 0,

	.2, 2.55, 0,		.4, 2.4, 0,					1.3, 2.4, 0,				1.3, 2.25, 0,
	.2, 2.55, .112,		.4, 2.4, .224,				1.3, 2.4, .728,				1.3, 2.25, .728,
	.112, 2.55, .2,		.224, 2.4, .4,				.728, 2.4, 1.3,				.728, 2.25, 1.3,
	0, 2.55, .2,		0, 2.4, .4,					0, 2.4, 1.3,				0, 2.25, 1.3,

	0, 2.55, .2,		0, 2.4, .4,					0, 2.4,	1.3,				0, 2.25, 1.3,
	-.112, 2.55, .2,	-.224, 2.4, .4,				-.728, 2.4, 1.3,			-.728, 2.25, 1.3,
	-.2, 2.55, .112,	-.4, 2.4, .224,				-1.3, 2.4, .728,			-1.3, 2.25, .728,
	-.2, 2.55, 0,		-.4, 2.4, 0,				-1.3, 2.4, 0,				-1.3, 2.25, 0,

	-.2, 2.55, 0,		-.4, 2.4, 0,				-1.3, 2.4, 0,				-1.3, 2.25, 0,
	-.2, 2.55, -.112,	-.4, 2.4, -.224,			-1.3, 2.4, -.728,			-1.3, 2.25, -.728,
	-.112, 2.55, -.2,	-.224, 2.4, -.4,			-.728, 2.4,	-1.3,			-.728, 2.25, -1.3,
	0, 2.55, -.2,		0, 2.4, -.4,				0, 2.4, -1.3,				0, 2.25, -1.3,

	0, 2.55, -.2,		0, 2.4, -.4,				0, 2.4, -1.3,				0, 2.25, -1.3,
	.112, 2.55, -.2,	.224, 2.4, -.4,				.728, 2.4, -1.3,			.728, 2.25, -1.3,
	.2, 2.55, -.112,	.4, 2.4, -.224,				1.3, 2.4, -.728,			1.3, 2.25, -.728,
	.2, 2.55, 0,		.4, 2.4, 0,					1.3, 2.4, 0,				1.3, 2.25, 0
#pragma endregion
};

B_Spline* teapot;


// Instantiates the teapot b-spline and sends the teapot control point data to it
void generateTeapot()
{
	Shader shader;
	shader.shaderPointer = shaderProgram;
	shader.uMPMat = uMPMat;
	shader.uMPVMat = uMPVMat;
	shader.uColor = uColor;

	teapot = new B_Spline(shader, 28);

	for (int i = 0; i < 28; ++i)
	{
		int k = i * 48;

		teapot->SetControlPoints(i,
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),

			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),

			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),

			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]),
			glm::vec3(teapotControlPoints[k++], teapotControlPoints[k++], teapotControlPoints[k++]));
	}

	teapot->transform().position = glm::vec3(0.0f, -1.5f, 0.0f);
}

void initShaders()
{
	char* shaders[] = { "fshader.glsl", "vshader.glsl" };
	GLenum types[] = { GL_FRAGMENT_SHADER, GL_VERTEX_SHADER };
	int numShaders = 2;
	
	shaderProgram = initShaders(shaders, types, numShaders);

	uMPMat = glGetUniformLocation(shaderProgram, "mpMat");
	uMPVMat = glGetUniformLocation(shaderProgram, "mpvMat");
	uColor = glGetUniformLocation(shaderProgram, "color");
}

void init()
{
	if (!glfwInit()) exit(EXIT_FAILURE);

	//Create window
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);

	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(800, 600, "Geometric_Lighting-GLFW", NULL, NULL); // Windowed

	//Activate window
	glfwMakeContextCurrent(window);

	glewExperimental = true;
	glewInit();

	initShaders();

	glfwSetTime(0.0);

	time_t timer;
	time(&timer);
	srand((unsigned int)timer);

	generateTeapot();

	InputManager::Init(window);
	CameraManager::Init(800.0f / 600.0f, 60.0f, 0.1f, 100.0f);

	glEnable(GL_DEPTH_TEST);
}

void step()
{
	// Clear to black
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	InputManager::Update();

	// Get delta time since the last frame
	float dt = (float)glfwGetTime();
	glfwSetTime(0.0);

	// Apply a rotation to the teapot if the user presses the right or left arrow keys
	float dTheta = 45.0f * InputManager::rightKey();
	dTheta -= 45.0f * InputManager::leftKey();

	teapot->transform().angularVelocity = glm::angleAxis(dTheta, glm::vec3(0.0f, 1.0f, 0.0f));

	// Update all components
	CameraManager::Update(dt);

	RenderManager::Update(dt);

	teapot->Update(dt);

	// Draw the display list
	RenderManager::Draw();

	// Swap buffers
	glfwSwapBuffers(window);
}

void cleanUp()
{
	glDeleteProgram(shaderProgram);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	RenderManager::DumpData();

	delete teapot;

	glfwTerminate();
}

int main()
{
	init();

	while (!glfwWindowShouldClose(window))
	{
		step();
		glfwPollEvents();
	}

	cleanUp();

	return 0;
}
