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

#include "solver_base.h"

//
template<typename U, typename R> struct HouseholderMethodsBase : SolverMethodsBase<U, R> {
	HouseholderMethodsBase (lua_State * L) : SolverMethodsBase<U, R>(L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(absDeterminant)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(logAbsDeterminant)
			}, {
				EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(solve)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};

//
template<typename U, typename R> struct AttachMethods<Eigen::HouseholderQR<U>, R> : HouseholderMethodsBase<Eigen::HouseholderQR<U>, R> {
	AttachMethods (lua_State * L) : HouseholderMethodsBase<Eigen::HouseholderQR<U>, R>(L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(householderQ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQR)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};

SOLVER_TYPE_NAME(HouseholderQR);

//
template<typename U, typename R> struct HouseholderExMethodsBase : HouseholderMethodsBase<U, R> {
	HouseholderExMethodsBase (lua_State * L) : HouseholderMethodsBase<U, R>(L)
	{
		HouseholderExtensions(L);
	}
};

//
template<typename U, typename R> struct AttachMethods<Eigen::ColPivHouseholderQR<U>, R> : HouseholderExMethodsBase<Eigen::ColPivHouseholderQR<U>, R> {
	AttachMethods (lua_State * L) : HouseholderExMethodsBase<Eigen::ColPivHouseholderQR<U>, R>(L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(colsPermutation)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(inverse)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQR)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixR)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};

SOLVER_TYPE_NAME(ColPivHouseholderQR);

//
template<typename U, typename R> struct AttachMethods<Eigen::CompleteOrthogonalDecomposition<U>, R> : HouseholderExMethodsBase<Eigen::CompleteOrthogonalDecomposition<U>, R> {
	AttachMethods (lua_State * L) : HouseholderExMethodsBase<Eigen::CompleteOrthogonalDecomposition<U>, R>(L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(householderQ)
			}, {
				"info", Info<>
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQTZ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixT)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixZ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(pseudoInverse)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};

SOLVER_TYPE_NAME(CompleteOrthogonalDecomposition);

//
template<typename U, typename R> struct AttachMethods<Eigen::FullPivHouseholderQR<U>, R> : HouseholderExMethodsBase<Eigen::FullPivHouseholderQR<U>, R> {
	AttachMethods (lua_State * L) : HouseholderExMethodsBase<Eigen::FullPivHouseholderQR<U>, R>(L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(inverse)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQ)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(matrixQR)
			}, {
				"rowsTranspositions", [](lua_State * L)
		{
			auto td = GetTypeData<Eigen::MatrixXi>(L);

			luaL_argcheck(L, td, 2, "rowsTranspositions() requires int matrices");

			Eigen::MatrixXi im = GetT(L)->rowsTranspositions();

			lua_getref(L, td->mPushRef);	// m, how, push
			lua_pushlightuserdata(L, &im);	// m, how, push, transps
			lua_call(L, 1, 1);	// m, how, transps_im

			return 1;
		}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};

SOLVER_TYPE_NAME(FullPivHouseholderQR);