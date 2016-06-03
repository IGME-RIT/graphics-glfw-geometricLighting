#pragma once
#include "RenderShape.h"

#include <GLEW\GL\glew.h>
#include <GLM\gtc\matrix_transform.hpp>
#include <vector>

class RenderShape;

class Patch
{
public:
	Patch(Shader shader);
	~Patch();

	void Update(float dt, bool updateSurface);

	void SetControlPoint(int controlPointIndex, glm::vec3 newPos);
	Transform& transform();
private:
	void UpdateSurface();
	void GeneratePlane();
	void AddVert(GLfloat x, GLfloat y, GLfloat z, GLfloat u, GLfloat v, int vertNum);
	void AddFace(GLint a, GLint b, GLint c, int faceNum);
private:
	glm::vec3 _controlPoints[16];
	RenderShape* _curve;
	GLuint _vao;
	GLuint _vbo;
	GLuint _ebo;

	Transform _transform;

	static const int NUM_VERTS = 20;
	static const int NUM_VERTS_STORED = NUM_VERTS * NUM_VERTS * 6;
	static const int NUM_ELEMENTS = (NUM_VERTS - 1) * (NUM_VERTS - 1) * 6;

	GLfloat _verts[NUM_VERTS_STORED];
	GLuint _elements[NUM_ELEMENTS];
};