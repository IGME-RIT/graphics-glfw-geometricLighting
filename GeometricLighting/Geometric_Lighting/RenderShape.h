#pragma once

#include <GLEW\GL\glew.h>
#include <GLM\gtc\matrix_transform.hpp>
#include <GLM\gtc\quaternion.hpp>
#include <GLM\gtc\type_ptr.hpp>

struct Transform
{
	glm::vec3 position;
	glm::vec3 rotationOrigin;
	glm::quat rotation;
	glm::vec3 scale;
	glm::vec3 scaleOrigin;

	glm::vec3 linearVelocity;
	glm::quat angularVelocity;

	glm::mat4 modelMat;

	Transform* parent;

	Transform()
	{
		position = glm::vec3();
		rotationOrigin = glm::vec3();
		rotation = glm::quat();
		scale = glm::vec3(1.0f, 1.0f, 1.0f);
		scaleOrigin = glm::vec3();

		linearVelocity = glm::vec3();
		angularVelocity = glm::quat();

		modelMat = glm::mat4();

		parent = (Transform*)nullptr;
	}
};

struct Shader
{
	GLint shaderPointer = 0;
	GLint uMPMat = 0;
	GLint uMPVMat = 0;
	GLint uColor = 0;
};

class RenderShape
{
public:
	RenderShape(GLint vao = 0, GLsizei count = 0, GLenum mode = 0, Shader shader = Shader(), glm::vec4 color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
	~RenderShape();

	void Update(float dt);
	void Draw();

	const glm::vec4& color();
	glm::vec4& currentColor();
	Transform& transform();
	GLint vao();
	GLsizei count();
	void count(GLsizei newSize);
	GLenum mode();
	Shader shader();
	bool& active();
	bool useDepthTest();

private:

	GLint _vao;
	GLsizei _count;
	GLenum _mode;
	Shader _shader;

protected:
	glm::vec4 _color;
	glm::vec4 _currentColor;
	Transform _transform;
	bool _active;
};
