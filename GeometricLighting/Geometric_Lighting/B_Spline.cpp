#include "B-Spline.h"
#include "Patch.h"

B_Spline::B_Spline(Shader shader, int numPatches)
{
	_spline = new std::vector<Patch*>();
	_spline->reserve(numPatches);

	for (int i = 0; i < numPatches; ++i)
	{
		(*_spline).push_back(new Patch(shader));
		(*_spline)[i]->transform().parent = &_transform;
	}

	_transform = Transform();
	_transform.position = glm::vec3();
	_transform.rotation = glm::quat();
	_transform.scale = glm::vec3(1.0f, 1.0f, 1.0f);

	_transform.angularVelocity = glm::quat();
	_transform.linearVelocity = glm::vec3();

	_transform.rotationOrigin = glm::vec3();
	_transform.scaleOrigin = glm::vec3();
}
B_Spline::~B_Spline()
{
	while (!_spline->empty())
	{
		Patch* toDelete = _spline->back();
		_spline->pop_back();
		delete toDelete;
	}
	delete _spline;
}

void B_Spline::Update(float dt)
{
	_transform.position += _transform.linearVelocity * dt;
	_transform.rotation = glm::slerp(_transform.rotation, _transform.rotation * _transform.angularVelocity, dt);

	glm::mat4 translateMat = glm::translate(glm::mat4(), _transform.position);

	glm::mat4 rotateOriginMat = glm::translate(glm::mat4(), _transform.rotationOrigin);
	glm::mat4 rotateMat = rotateOriginMat * glm::mat4_cast(_transform.rotation) * glm::inverse(rotateOriginMat);

	glm::mat4 scaleOriginMat = glm::translate(glm::mat4(), _transform.scaleOrigin);
	glm::mat4 scaleMat = scaleOriginMat * glm::scale(glm::mat4(), _transform.scale) * glm::inverse(scaleOriginMat);

	glm::mat4 *parentModelMat = _transform.parent ? &_transform.parent->modelMat : &glm::mat4();

	_transform.modelMat = (*parentModelMat) * (translateMat * scaleMat* rotateMat);

	unsigned int size = _spline->size();
	for (unsigned int i = 0; i < size; ++i)
	{
		(*_spline)[i]->Update(dt, false);
	}
}

void B_Spline::SetControlPoints(int patch,
	glm::vec3 controlPointPos0, glm::vec3 controlPointPos1, glm::vec3 controlPointPos2, glm::vec3 controlPointPos3,
	glm::vec3 controlPointPos4, glm::vec3 controlPointPos5, glm::vec3 controlPointPos6, glm::vec3 controlPointPos7,
	glm::vec3 controlPointPos8, glm::vec3 controlPointPos9, glm::vec3 controlPointPos10, glm::vec3 controlPointPos11,
	glm::vec3 controlPointPos12, glm::vec3 controlPointPos13, glm::vec3 controlPointPos14, glm::vec3 controlPointPos15)
{
	(*_spline)[patch]->SetControlPoint(0, controlPointPos0);
	(*_spline)[patch]->SetControlPoint(1, controlPointPos1);
	(*_spline)[patch]->SetControlPoint(2, controlPointPos2);
	(*_spline)[patch]->SetControlPoint(3, controlPointPos3);

	(*_spline)[patch]->SetControlPoint(4, controlPointPos4);
	(*_spline)[patch]->SetControlPoint(5, controlPointPos5);
	(*_spline)[patch]->SetControlPoint(6, controlPointPos6);
	(*_spline)[patch]->SetControlPoint(7, controlPointPos7);

	(*_spline)[patch]->SetControlPoint(8, controlPointPos8);
	(*_spline)[patch]->SetControlPoint(9, controlPointPos9);
	(*_spline)[patch]->SetControlPoint(10, controlPointPos10);
	(*_spline)[patch]->SetControlPoint(11, controlPointPos11);

	(*_spline)[patch]->SetControlPoint(12, controlPointPos12);
	(*_spline)[patch]->SetControlPoint(13, controlPointPos13);
	(*_spline)[patch]->SetControlPoint(14, controlPointPos14);
	(*_spline)[patch]->SetControlPoint(15, controlPointPos15);
	(*_spline)[patch]->Update(0.0f, true);
}

Transform& B_Spline::transform() { return _transform; }