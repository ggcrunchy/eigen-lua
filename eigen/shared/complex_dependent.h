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
#include "utils/LuaEx.h"
#include "types.h"
#include "utils.h"
#include "macros.h"

// Get a component from a complex matrix.
// TODO: these actually implement a UnaryView, I think...
#define EIGEN_COMPONENT_GET(METHOD)	using RM = MatrixOf<T::Scalar::value_type>;				\
																							\
									auto td = TypeData<RM>::Get(L);							\
																							\
									luaL_argcheck(L, td, 1, "Real matrix type unavailable");\
																							\
									RM m = GetT(L)->METHOD();								\
																							\
									PUSH_TYPED_DATA(m)

// Assign a component in a complex matrix.
// TODO: see note for GET()... if that was implemented, should instead be method of that
#define EIGEN_COMPONENT_ASSIGN(METHOD, REAL_TYPE)	using Real = REAL_TYPE;																							\
																																									\
													T & m = *GetT(L);																								\
																																									\
													if (HasType<Eigen::MatrixXi>(L, 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXi>(L, 2)->cast<Real>();				\
													else if (HasType<Eigen::MatrixXf>(L, 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXf>(L, 2)->cast<Real>();			\
													else if (HasType<Eigen::MatrixXd>(L, 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXd>(L, 2)->cast<Real>();			\
													else if (HasType<Eigen::MatrixXcf>(L, 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXcf>(L, 2)->METHOD().cast<Real>();\
													else if (HasType<Eigen::MatrixXcd>(L, 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXcd>(L, 2)->METHOD().cast<Real>();\
													else luaL_error(L, "Unsupported type");																			\
																																									\
													return 0

#define EIGEN_COMPONENT_GET_METHOD(NAME) EIGEN_REG(NAME, EIGEN_COMPONENT_GET(NAME))
#define EIGEN_COMPONENT_ASSIGN_METHOD(NAME, REAL_TYPE) EIGEN_REG(NAME "Assign", EIGEN_COMPONENT_ASSIGN(NAME, REAL_TYPE))

// Methods assigned when the matrix is complex.
template<typename T, typename R, bool = Eigen::NumTraits<T::Scalar>::IsComplex> struct ComplexDependentMethods : InstanceGetters<T, R> {
	ComplexDependentMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_COMPONENT_GET_METHOD(imag)
			}, {
				EIGEN_COMPONENT_ASSIGN_METHOD(imag, T::Scalar::value_type)
			}, {
				EIGEN_COMPONENT_GET_METHOD(real)
			}, {
				EIGEN_COMPONENT_ASSIGN_METHOD(real, T::Scalar::value_type)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};

// Methods assigned when the matrix is real.
template<typename T, typename R> struct ComplexDependentMethods<T, R, false> : InstanceGetters<T, R> {
	// Version of methods when we have a matrix or basic map.
	template<bool = HasNormalStride<T>::value> void AddIfNormalStride (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(asPermutation)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	// Version of methods with maps having non-trivial stride.
	template<> void AddIfNormalStride<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"asPermutation", [](lua_State * L)
				{
					R temp = *GetT(L);

					return NewRet<R>(L, temp.asPermutation());
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
		
	ComplexDependentMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_ARRAY_METHOD(ceil)
			}, {
				EIGEN_REL_OP_METHOD(cwiseGreaterThan, >)
			}, {
				EIGEN_REL_OP_METHOD(cwiseGreaterThanOrEqual, >=)
			}, {
				EIGEN_REL_OP_METHOD(cwiseLessThan, <)
			}, {
				EIGEN_REL_OP_METHOD(cwiseLessThanOrEqual, <=)
			}, {
				EIGEN_MATRIX_GET_MATRIX_SECOND_IS_MATRIX_OR_SCALAR_METHOD(cwiseMax)
			}, {
				EIGEN_MATRIX_GET_MATRIX_SECOND_IS_MATRIX_OR_SCALAR_METHOD(cwiseMin)
			}, {
				EIGEN_ARRAY_METHOD(floor)
			}, {
				"imag", [](lua_State * L)
				{
					return NewRet<R>(L, R{});
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(maxCoeff)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(minCoeff)
			}, {
				"real", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L));
				}
			}, {
				EIGEN_COMPONENT_ASSIGN_METHOD(real, T::Scalar)
			}, {
				EIGEN_ARRAY_METHOD(round)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		AddIfNormalStride(L);
	}
};