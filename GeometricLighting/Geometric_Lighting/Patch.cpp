#include "Patch.h"
#include "RenderManager.h"
#include "RenderShape.h"
#include "Init_Shader.h"
#include "InputManager.h"

#include <vector>

Patch::Patch(Shader shader)
{
	_transform = Transform();
	_transform.position = glm::vec3();
	_transform.rotation = glm::quat();
	_transform.scale = glm::vec3(1.0f, 1.0f, 1.0f);

	_transform.angularVelocity = glm::quat();
	_transform.linearVelocity = glm::vec3();

	_transform.rotationOrigin = glm::vec3();
	_transform.scaleOrigin = glm::vec3();

	GLfloat data = 0.0f;
	GLint elements = 0;

	glGenVertexArrays(1, &_vao);
	glBindVertexArray(_vao);

	glGenBuffers(1, &_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * NUM_VERTS_STORED, &data, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &_ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLint) * NUM_ELEMENTS, &elements, GL_DYNAMIC_DRAW);

	// Bind buffer data to shader values
	GLint posAttrib = glGetAttribLocation(shader.shaderPointer, "position");
	glEnableVertexAttribArray(posAttrib);
	glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), 0);

	GLint normAttrib = glGetAttribLocation(shader.shaderPointer, "normal");
	glEnableVertexAttribArray(normAttrib);
	glVertexAttribPointer(normAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

	_curve = new RenderShape(_vao, NUM_ELEMENTS, GL_TRIANGLES, shader, glm::vec4(0.6f, 0.6f, 0.6f, 1.0f));

	_curve->transform().parent = &_transform;

	RenderManager::AddShape(_curve);

	GeneratePlane();
}
Patch::~Patch()
{
	glDeleteBuffers(1, &_vbo);
	glDeleteVertexArrays(1, &_vao);
	glDeleteBuffers(1, &_ebo);
}

void Patch::Update(float dt, bool updateSurface)
{
	if (updateSurface)
		UpdateSurface();

	_transform.position += _transform.linearVelocity * dt;
	_transform.rotation = glm::slerp(_transform.rotation, _transform.rotation * _transform.angularVelocity, dt);

	glm::mat4 translateMat = glm::translate(glm::mat4(), _transform.position);

	glm::mat4 rotateOriginMat = glm::translate(glm::mat4(), _transform.rotationOrigin);
	glm::mat4 rotateMat = rotateOriginMat * glm::mat4_cast(_transform.rotation) * glm::inverse(rotateOriginMat);

	glm::mat4 scaleOriginMat = glm::translate(glm::mat4(), _transform.scaleOrigin);
	glm::mat4 scaleMat = scaleOriginMat * glm::scale(glm::mat4(), _transform.scale) * glm::inverse(scaleOriginMat);

	glm::mat4 *parentModelMat = _transform.parent ? &_transform.parent->modelMat : &glm::mat4();

	_transform.modelMat = (*parentModelMat) * (translateMat * scaleMat* rotateMat);
}

void Patch::SetControlPoint(int controlPointIndex, glm::vec3 newPos)
{
	_controlPoints[controlPointIndex] = newPos;
}

Transform& Patch::transform() { return _transform; }

void Patch::UpdateSurface()
{
	GLfloat  inc = 1.0f / ((float)NUM_VERTS - 1.0f);
	GLfloat t = 0.0f;

	GLfloat factors[NUM_VERTS][7];

	for (int i = 0; i < NUM_VERTS; ++i, t += inc)
	{
		GLfloat t_sqr = t * t;
		GLfloat t_inv = (1 - t);
		GLfloat t_inv_sqr = t_inv * t_inv;

		// These are the factors used in a Bernstein polynomial
		// Bernstein polynomials increase in order as the number of 
		// control points increases. For four control points, we'll
		// use a third order polynomial. 
		factors[i][0] = t_inv * t_inv_sqr;
		factors[i][1] = 3 * t * t_inv_sqr;
		factors[i][2] = 3 * t_sqr * t_inv;
		factors[i][3] = t * t_sqr;

		// One of the great mathematical properties of Bernstein 
		// polynomials is that each order lower you go, you go 
		// down one derivation and each order higher you go, you
		// go up one integration. Calculus!
		// In this case, we're storing one order lower to get 
		// the slope of the bezier curve for finding the normal
		factors[i][4] = t_inv_sqr;
		factors[i][5] = 2 * t * t_inv;
		factors[i][6] = t_sqr;
	}

	glm::vec3 newControlPoints[4];
	glm::vec3 newSlopeControlPoints[4];
	for (int i = 0; i < NUM_VERTS; ++i)
	{
		newControlPoints[0] = factors[i][0] * _controlPoints[0] + factors[i][1] * _controlPoints[1] + factors[i][2] * _controlPoints[2] + factors[i][3] * _controlPoints[3];
		newControlPoints[1] = factors[i][0] * _controlPoints[4] + factors[i][1] * _controlPoints[5] + factors[i][2] * _controlPoints[6] + factors[i][3] * _controlPoints[7];
		newControlPoints[2] = factors[i][0] * _controlPoints[8] + factors[i][1] * _controlPoints[9] + factors[i][2] * _controlPoints[10] + factors[i][3] * _controlPoints[11];
		newControlPoints[3] = factors[i][0] * _controlPoints[12] + factors[i][1] * _controlPoints[13] + factors[i][2] * _controlPoints[14] + factors[i][3] * _controlPoints[15];

		// These represent the columnar tangents for each row of verticies in the bezier surface
		// Use the derivitave of the Bernstein polynomial to determine the normal of the curve at this point using the difference between control points as slope control points
		// (1-t)^2 + 2t(1-t) + t^2
		newSlopeControlPoints[0] = factors[i][4] * (_controlPoints[1] - _controlPoints[0]) + factors[i][5] * (_controlPoints[2] - _controlPoints[1]) + factors[i][6] * (_controlPoints[3] - _controlPoints[2]);
		newSlopeControlPoints[1] = factors[i][4] * (_controlPoints[5] - _controlPoints[4]) + factors[i][5] * (_controlPoints[6] - _controlPoints[5]) + factors[i][6] * (_controlPoints[7] - _controlPoints[6]);
		newSlopeControlPoints[2] = factors[i][4] * (_controlPoints[9] - _controlPoints[8]) + factors[i][5] * (_controlPoints[10] - _controlPoints[9]) + factors[i][6] * (_controlPoints[11] - _controlPoints[10]);
		newSlopeControlPoints[3] = factors[i][4] * (_controlPoints[13] - _controlPoints[12]) + factors[i][5] * (_controlPoints[14] - _controlPoints[13]) + factors[i][6] * (_controlPoints[15] - _controlPoints[14]);

		for (int j = 0; j < NUM_VERTS; ++j)
		{
			glm::vec3 newPoint = factors[j][0] * newControlPoints[0] + factors[j][1] * newControlPoints[1] + factors[j][2] * newControlPoints[2] + factors[j][3] * newControlPoints[3];
			_verts[(j + (i * NUM_VERTS)) * 6] = newPoint.x;
			_verts[(j + (i * NUM_VERTS)) * 6 + 1] = newPoint.y;
			_verts[(j + (i * NUM_VERTS)) * 6 + 2] = newPoint.z;

			// This tangent represents the row tangent, so the tangent of the surface relative to the surface's x direction
			glm::vec3 tangentA = factors[j][4] * (newControlPoints[1] - newControlPoints[0]) + factors[j][5] * (newControlPoints[2] - newControlPoints[1]) + factors[j][6] * (newControlPoints[3] - newControlPoints[2]);

			// This tangent is a second valid tangent necessary for finding a cross product, this one being relative to the surface's y direction
			glm::vec3 tangentB = factors[j][0] * newSlopeControlPoints[0] + factors[j][1] * newSlopeControlPoints[1] + factors[j][2] * newSlopeControlPoints[2] + factors[j][3] * newSlopeControlPoints[3];

			// By taking the normal of these two tangents, we can get the normal to the surface
			glm::vec3 normal = glm::cross(glm::normalize(tangentB), glm::normalize(tangentA));

			_verts[(j + (i * NUM_VERTS)) * 6 + 3] = normal.x;
			_verts[(j + (i * NUM_VERTS)) * 6 + 4] = normal.y;
			_verts[(j + (i * NUM_VERTS)) * 6 + 5] = normal.z;
		}
	}
	glBindBuffer(GL_VERTEX_ARRAY, _vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(_verts), (void*)&_verts, GL_DYNAMIC_DRAW);
}

void Patch::GeneratePlane()
{
	// Allocate vertices for the plane
	int vertNum = 0;
	GLfloat numVertsf = (GLfloat)NUM_VERTS;
	for (GLfloat i = 0.0f; i < numVertsf; i += 1.0f)
	{
		for (GLfloat j = 0.0f; j < numVertsf; j += 1.0f)
		{
			AddVert(0.0f, 0.0f, 0.0f, i / (numVertsf - 1.0f), j / (numVertsf - 1.0f), vertNum++);
		}
	}

	int cp = 0;
	float zOffset = 1.0f / 3.0f;
	float xOffset = 1.0f / 3.0f;
	glm::vec3 baseVec = glm::vec3(-0.5f, 0.0f, -0.5f);
	for (int row = 0; row < 4; ++row)
	{
		_controlPoints[cp++] = glm::vec3(baseVec.x, baseVec.y, baseVec.z + zOffset * row);
		_controlPoints[cp++] = glm::vec3(baseVec.x + xOffset, baseVec.y, baseVec.z + zOffset * row);
		_controlPoints[cp++] = glm::vec3(baseVec.x + xOffset * 2, baseVec.y, baseVec.z + zOffset * row);
		_controlPoints[cp++] = glm::vec3(baseVec.x + xOffset * 3, baseVec.y, baseVec.z + zOffset * row);
	}

	// Add elements for faces
	int faceNum = 0;
	int quadsPerRow = NUM_VERTS * (NUM_VERTS - 1);
	for (int i = 0; i < quadsPerRow; i += NUM_VERTS)
	{
		for (int j = 0; j < NUM_VERTS - 1; ++j)
		{
			AddFace(i + j, i + j + 1, i + NUM_VERTS + j, faceNum++);
			AddFace(i + j + 1, i + NUM_VERTS + j + 1, i + NUM_VERTS + j, faceNum++);
		}
	}
	glBindBuffer(GL_VERTEX_ARRAY, _vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(_elements), (void*)&_elements, GL_DYNAMIC_DRAW);

	UpdateSurface();
}

void Patch::AddVert(GLfloat x, GLfloat y, GLfloat z, GLfloat u, GLfloat v, int vertNum)
{
	int itr = vertNum * 6;
	_verts[itr++] = x;
	_verts[itr++] = y;
	_verts[itr++] = z;
	_verts[itr++] = 0.0f;
	_verts[itr++] = 1.0f;
	_verts[itr] = 0.0f;
}

void Patch::AddFace(GLint a, GLint b, GLint c, int faceNum)
{
	_elements[faceNum * 3] = a;
	_elements[faceNum * 3 + 1] = b;
	_elements[faceNum * 3 + 2] = c;
}
