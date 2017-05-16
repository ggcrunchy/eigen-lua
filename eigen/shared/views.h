/*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* [ MIT license: http://www.opensource.org/licenses/mit-license.php ]
*/

#pragma once

#include "CoronaLua.h"
#include "macros.h"
#include "types.h"
#include "utils.h"
#include "views.h"
#include <Eigen/Eigen>
#include <type_traits>
#include <utility>

#define ENUM_PAIR(ENUM)	ENUM, ENUM##SA
#define ENUM_EXTRAS(ENUM)	ENUM##_Adjoint,		\
							ENUM##_Conjugate,	\
							ENUM##_CwiseAdjoint

#define METHOD_PAIR(ENUM, BASE)	DO_METHOD(ENUM, BASE)		\
								DO_METHOD(ENUM##SA, BASE)
#define EXTRA_METHODS(ENUM, BASE)	DO_METHOD(ENUM##_Adjoint, BASE)		\
									DO_METHOD(ENUM##_Conjugate, BASE)	\
									DO_METHOD(ENUM##_CwiseAdjoint, BASE)

#define AUX_VIEW_ENUMS(ENUM, SUFFIX)	ENUM##_##SUFFIX,						\
										ENUM_PAIR(ENUM##_Const##SUFFIX),		\
										ENUM_PAIR(ENUM##_Trans##SUFFIX),		\
										ENUM_PAIR(ENUM##_ConstTrans##SUFFIX),	\
										ENUM_EXTRAS(ENUM##_##SUFFIX)
#define AUX_VIEW_METHODS(ENUM, SUFFIX)	DO_METHOD(ENUM##_##SUFFIX, ENUM)				\
										METHOD_PAIR(ENUM##_Const##SUFFIX, ENUM)			\
										METHOD_PAIR(ENUM##_Trans##SUFFIX, ENUM)			\
										METHOD_PAIR(ENUM##_ConstTrans##SUFFIX, ENUM)	\
										EXTRA_METHODS(ENUM##_##SUFFIX, ENUM)
#ifdef WANT_MAP
	#define VIEW_ENUMS(ENUM)	ENUM,											\
								AUX_VIEW_ENUMS(ENUM, Basic),					\
								AUX_VIEW_ENUMS(ENUM, Mapped),					\
								AUX_VIEW_ENUMS(ENUM, MappedWithInnerStride),	\
								AUX_VIEW_ENUMS(ENUM, MappedWithOuterStride)
	#define VIEW_METHODS(ENUM)	AUX_VIEW_METHODS(ENUM, Basic)					\
								AUX_VIEW_METHODS(ENUM, Mapped)					\
								AUX_VIEW_METHODS(ENUM, MappedWithInnerStride)	\
								AUX_VIEW_METHODS(ENUM, MappedWithOuterStride)
#else
	#define VIEW_ENUMS(ENUM)	ENUM,						\
								AUX_VIEW_ENUMS(ENUM, Basic)
	#define VIEW_METHODS(ENUM)	AUX_VIEW_METHODS(ENUM, Basic)
#endif

#define ADD_VIEW_PAIR(VIEW, ENUM, EX, T, NAME)	ADD_OBJECT_TYPE_ID(WRAP2(VIEW, T, EX), ENUM, NAME);									\
												ADD_OBJECT_TYPE_ID(WRAP2(VIEW, T, EX | Eigen::SelfAdjoint), ENUM##SA, NAME "sa")

#define AUX_UNROLL_VIEW(VIEW, ENUM, EX, T, NAME, SUFFIX)	ADD_OBJECT_TYPE_ID(WRAP2(VIEW, Unmapped<T>::SUFFIX##Type, EX), ENUM##_##SUFFIX, NAME);				\
															ADD_COMPLEX_TYPES(VIEW, ENUM##_##SUFFIX, EX, Unmapped<T>::SUFFIX##Type, NAME);						\
															ADD_VIEW_PAIR(VIEW, ENUM##_Const##SUFFIX, EX, const Unmapped<T>::SUFFIX##Type, NAME "k");			\
															ADD_VIEW_PAIR(VIEW, ENUM##_Trans##SUFFIX, EX, Unmapped<T>::Trans##SUFFIX##Type, NAME "t");			\
															ADD_VIEW_PAIR(VIEW, ENUM##_ConstTrans##SUFFIX, EX, Unmapped<T>::ConstTrans##SUFFIX##Type, NAME "kt")

#define AUX_ADD_COMPLEX_VIEW_TYPES(VIEW, ENUM, EX, T, NAME)	ADD_OBJECT_TYPE_ID(WRAP2(VIEW, T, EX)::AdjointReturnType, ENUM##_Adjoint, NAME "_art");			\
															ADD_OBJECT_TYPE_ID(WRAP2(VIEW, T, EX)::ConjugateReturnType, ENUM##_Conjugate, NAME "_crt");		\
															ADD_OBJECT_TYPE_ID(WRAP2(VIEW, T::AdjointReturnType, EX), ENUM##_CwiseAdjoint, NAME "_cwart")

#ifdef WANT_MAP
	#define ADD_COMPLEX_VIEW_TYPES(VIEW, ENUM, EX, T, NAME)	AUX_ADD_COMPLEX_VIEW_TYPES(VIEW, ENUM, EX, T, NAME);																\
															AUX_ADD_COMPLEX_VIEW_TYPES(VIEW, ENUM##_Mapped, EX, Unmapped<T>::MappedType, NAME);									\
															AUX_ADD_COMPLEX_VIEW_TYPES(VIEW, ENUM##_MappedWithInnerStride, EX, Unmapped<T>::MappedWithInnerStrideType, NAME);	\
															AUX_ADD_COMPLEX_VIEW_TYPES(VIEW, ENUM##_MappedWithOuterStride, EX, Unmapped<T>::MappedWithOuterStrideType, NAME)
#else
	#define ADD_COMPLEX_VIEW_TYPES AUX_ADD_COMPLEX_VIEW_TYPES
#endif

#ifdef WANT_MAP
	#define UNROLL_VIEW(VIEW, ENUM, EX, T, NAME)	AUX_UNROLL_VIEW(VIEW, ENUM, EX, T, NAME, Basic);						\
													AUX_UNROLL_VIEW(VIEW, ENUM, EX, T, NAME "_vm", Mapped);					\
													AUX_UNROLL_VIEW(VIEW, ENUM, EX, T, NAME "_vmis", MappedWithInnerStride);\
													AUX_UNROLL_VIEW(VIEW, ENUM, EX, T, NAME "_vmos", MappedWithOuterStride)
#else
	#define UNROLL_VIEW(VIEW, ENUM, EX, T, NAME)	AUX_UNROLL_VIEW(VIEW, ENUM, EX, T, NAME, Basic)
#endif

