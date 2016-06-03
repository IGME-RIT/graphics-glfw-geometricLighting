#version 440

in vec3 position;
in vec3 normal;

uniform mat4 mpvMat;
uniform mat4 mpMat;
uniform vec4 color;

out vec4 Color;
out vec4 Normal;
out vec4 WorldPos;

void main()
{
	Color = color;
	Normal = mpMat * vec4(normal, 0.0);
	WorldPos = mpvMat * vec4(position, 1.0);
	gl_Position = WorldPos;
}