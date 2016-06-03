#pragma once

#include <GLEW\GL\glew.h>

static char* textFileRead(char* fn);

GLuint initShaders(char** shaders, GLenum* types, int numShaders);


