#pragma once
#include "RenderShape.h"

#include <vector>

class Patch;
class RenderShape;

class B_Spline
{
public:
	B_Spline(Shader shader, int numPatches = 1);
	~B_Spline();

	void Update(float dt);
	void SetControlPoints(int patch,
		glm::vec3 controlPointPos0, glm::vec3 controlPointPos1, glm::vec3 controlPointPos2, glm::vec3 controlPointPos3,
		glm::vec3 controlPointPos4, glm::vec3 controlPointPos5, glm::vec3 controlPointPos6, glm::vec3 controlPointPos7,
		glm::vec3 controlPointPos8, glm::vec3 controlPointPos9, glm::vec3 controlPointPos10, glm::vec3 controlPointPos11,
		glm::vec3 controlPointPos12, glm::vec3 controlPointPos13, glm::vec3 controlPointPos14, glm::vec3 controlPointPos15);

	Transform& transform(); 
private:
	Transform _transform;

	std::vector<Patch*>* _spline;
};