#version 440

in vec4 Color;
in vec4 Normal;
in vec4 WorldPos;

out vec4 outColor;

void main()
{
	vec4 lightPos = vec4(8.0, 0.0, 0.0, 1.0);
	vec4 lightDir = normalize(WorldPos - lightPos);
	
	float diffusePower = 5.0;

	vec4 lightDiffuseColor = vec4(1.0, 1.0, 1.0, 1.0);
	vec4 ambient = vec4(0.3, 0.3, 0.3, 1.0);
	
	float dis = length(lightDir);
	
	float NdotL = dot(Normal, lightDir);
	float intensity = clamp(NdotL, 0.0, 1.0);
	vec4 diffuse = intensity * lightDiffuseColor * diffusePower / (dis * dis);
	
	outColor = (diffuse + ambient) * Color;
};