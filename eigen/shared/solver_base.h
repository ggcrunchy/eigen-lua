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
#include <Eigen/Eigen>
#include <utility>

//
#define SOLVER_TYPE_NAME(TYPE)	template<typename T> struct AuxTypeName<Eigen::TYPE<T>> {	\
									AuxTypeName (luaL_Buffer * B, lua_State * L)			\
									{														\
										luaL_addstring(B, #TYPE "<");						\
																							\
										AuxTypeName<T>(B, L);								\
																							\
										luaL_addstring(B, ">");								\
									}														\
								}

//
#define SOLVER_TYPE_NAME_EX(TYPE)	template<typename T, int EX> struct AuxTypeName<Eigen::TYPE<T, EX>> {	\
										AuxTypeName (luaL_Buffer * B, lua_State * L)						\
										{																	\
											luaL_addstring(B, #TYPE "<");									\
																											\
											AuxTypeName<T>(B, L);											\
																											\
											lua_pushfstring(L, ", %d>", EX); /* ..., ex */					\
											luaL_addvalue(B);	/* ... */									\
										}																	\
									}

//
template<typename T, typename R> struct SolverMethodsBase {
	ADD_INSTANCE_GETTERS()

	//
	using Real = typename Eigen::NumTraits<typename R::Scalar>::Real;

	//
	template<bool = true> static int Info (lua_State * L) // dummy template parameter as lazy enable_if
	{
		switch (GetT(L)->info())
		{
		case Eigen::Success:
			lua_pushliteral(L, "success");	// ..., "success"
			break;
		case Eigen::NumericalIssue:
			lua_pushliteral(L, "numerical_issue");	// ..., "numerical_issue"
			break;
		default:	// Errors trapped by asserts
			lua_pushliteral(L, "no_convergence");	// ..., "no_convergence"
		}

		return 1;
	}

	//
	template<bool = true> static int SetMaxIterations (lua_State * L)
	{
		GetT(L)->setMaxIterations(LuaXS::Int(L, 2));

		return SelfForChaining(L);	// solver, count, solver
	}

	//
	template<bool = true> static int SetThreshold (lua_State * L)
	{
		lua_settop(L, 2);	// solver, ..., how
		lua_pushliteral(L, "default");	// solver, ..., how, "default"

		if (!lua_equal(L, 2, 3)) GetT(L)->setThreshold(LuaXS::GetArg<Real>(L, 2));

		else GetT(L)->setThreshold(Eigen::Default_t{});

		return SelfForChaining(L);	// solver, ..., how, "default", solver
	}

	template<bool = true> void HouseholderExtensions (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isInjective)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isInvertible)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(isSurjective)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(maxPivot)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(nonzeroPivots)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rank)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(threshold)
			}, {
				"setThreshold", SetThreshold<>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	SolverMethodsBase (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"__gc", LuaXS::TypedGC<T>
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};