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

//
template<typename T, typename R> struct ComplexDependentMethods {
	//
	ADD_INSTANCE_GETTERS()

	//
	ComplexDependentMethods (lua_State * L)
	{
		Add(L);
	}

	#define EIGEN_COMPONENT_GET(METHOD)	using RM = Eigen::Matrix<T::Scalar::value_type, Eigen::Dynamic, Eigen::Dynamic>;\
																														\
										auto td = GetTypeData<RM>(L);													\
																														\
										luaL_argcheck(L, td, 1, "Real matrix type unavailable");						\
																														\
										RM m = GetT(L)->METHOD();														\
																														\
										lua_getref(L, td->mPushRef);/* mat, push_type */								\
										lua_pushlightuserdata(L, &m);	/* mat, push_type, comp */						\
										lua_call(L, 1, 1);	/* mat, conv_mat */											\
																														\
										return 1

	#define EIGEN_COMPONENT_ASSIGN(METHOD, REAL_TYPE)	using Real = REAL_TYPE;																												\
																																															\
														T & m = *GetT(L);																													\
																																															\
														if (LuaXS::IsType(L, FullName<Eigen::MatrixXi>(), 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXi>(L, 2)->cast<Real>();					\
														else if (LuaXS::IsType(L, FullName<Eigen::MatrixXf>(), 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXf>(L, 2)->cast<Real>();				\
														else if (LuaXS::IsType(L, FullName<Eigen::MatrixXd>(), 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXd>(L, 2)->cast<Real>();				\
														else if (LuaXS::IsType(L, FullName<Eigen::MatrixXcf>(), 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXcf>(L, 2)->METHOD().cast<Real>();	\
														else if (LuaXS::IsType(L, FullName<Eigen::MatrixXcd>(), 2)) m.METHOD() = LuaXS::UD<Eigen::MatrixXcd>(L, 2)->METHOD().cast<Real>();	\
														else luaL_error(L, "Unsupported type");																								\
																																															\
														return 0

	#define EIGEN_COMPONENT_GET_METHOD(NAME) EIGEN_REG(NAME, EIGEN_COMPONENT_GET(NAME))
	#define EIGEN_COMPONENT_ASSIGN_METHOD(NAME, REAL_TYPE) EIGEN_REG(NAME "Assign", EIGEN_COMPONENT_ASSIGN(NAME, REAL_TYPE))

	//
	template<bool = Eigen::NumTraits<T::Scalar>::IsComplex> void Add (lua_State * L)
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

	//
	template<bool = std::is_same<T, R>::value
#ifdef WANT_MAP
		|| std::is_same<T, Unmapped<T>::MappedType>::value
#endif
	> void AddIfNormalStride (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(asPermutation)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	template<> void AddIfNormalStride<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"asPermutation", [](lua_State * L)
				{
					R temp = *GetT(L);

					return NewMoveRet<R>(L, temp.asPermutation());
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	template<> void Add<false> (lua_State * L)
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
					return NewMoveRet<R>(L, R{});
				}
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(maxCoeff)
			}, {
				EIGEN_MATRIX_REDUCE_METHOD(minCoeff)
			}, {
				"real", [](lua_State * L)
				{
					return NewMoveRet<R>(L, *GetT(L));
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