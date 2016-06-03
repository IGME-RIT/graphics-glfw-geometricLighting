#pragma once
#include <GLEW\GL\glew.h>
#include <GLM\gtc\matrix_transform.hpp>
#include <vector>

struct Transform;
struct Shader;
class RenderShape;

class RenderManager
{
public:
	static void AddShape(Shader shader, GLuint vao, GLenum type, GLsizei count, glm::vec4 color, Transform transform);
	
	static void AddShape(RenderShape* shape);

	static void Update(float dt);

	static void Draw();

	static void DumpData();

private:

	static std::vector<RenderShape*> _shapes;
};
