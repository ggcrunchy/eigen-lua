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
#include "types.h"
#include "utils.h"
#include "macros.h"
#include <type_traits>

//
template<typename T, typename R> struct MapDependentMethods {
	//
	ADD_INSTANCE_GETTERS()

	//
	MapDependentMethods (lua_State * L)
	{
		NoMap(L);
	}

	//
	template<bool = !std::is_same<T, R>::value> void NoMap (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"__add", [](lua_State * L)
				{
					return NewMoveRet<R>(L, *GetT(L) + *GetR(L, 2));
				}
			}, {
				"__sub", [](lua_State * L)
				{
					return NewMoveRet<R>(L, *GetT(L) - *GetR(L, 2));
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	struct ResizeState {
		int mDim1{1}, mDim2{1};
		bool mHas1{true}, mHas2{true};

		ResizeState (lua_State * L, const T & mat)
		{
			if (!lua_isnoneornil(L, 2))
			{
				lua_pushliteral(L, "no_change");// mat, a, b, "no_change"

				mHas1 = lua_equal(L, 2, -1) != 0;
				mHas2 = lua_equal(L, 3, -1) != 0;

				luaL_argcheck(L, mHas1 || mHas2, 1, "Must resize at least one dimension");

				if (mHas1) mDim1 = luaL_checkint(L, 2);
				if (mHas2) mDim2 = luaL_checkint(L, 3);
			}
		
			else
			{
				CheckVector(L, mat, 1);

				int a = luaL_checkint(L, 2);

				if (mat.cols() == 1) mDim2 = a;

				else mDim1 = a;
			}
		}
	};

	#define EIGEN_MATRIX_RESIZE(METHOD)	T & m = *GetT(L);										\
										ResizeState rs{L, m};									\
																								\
										if (!rs.mHas2) m.METHOD(rs.mDim1, Eigen::NoChange);		\
										else if (!rs.mHas1) m.METHOD(Eigen::NoChange, rs.mDim2);\
										else m.METHOD(rs.mDim1, rs.mDim2);						\
																								\
										return 0
	#define EIGEN_MATRIX_RESIZE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_RESIZE(NAME))

	//
	static bool GetEigenvectors (lua_State * L)
	{
		lua_settop(L, 2);	// mat, ignore?
		lua_pushliteral(L, "no_eigenvectors");	// mat, ignore?, "no_eigenvectors"

		return lua_equal(L, 2, 3) == 0;
	}

	//
	template<bool = std::is_same<T::Scalar, int>::value, bool = !std::is_arithmetic<T::Scalar>::value> void IgnoreWhenComplex (lua_State * L) {}

	//
	template<> void IgnoreWhenComplex<false, true> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"eigenvalues", [](lua_State * L)
				{
					bool bGetVectors = GetEigenvectors(L);	// mat, ignore?, "no_eigenvectors"

					Eigen::ComplexEigenSolver<T> es{*GetT(L), bGetVectors};

					int ret = NewMoveRet<R>(L, es.eigenvalues());	// mat, ignore?, "no_eigenvectors", values

					if (bGetVectors) ret += NewMoveRet<R>(L, es.eigenvectors());// mat, ignore?, "no_eigenvectors", values, vectors

					return ret;
				}
			},
			{ nullptr, nullptr }
		};
		
		luaL_register(L, nullptr, methods);
	}

	template<> void IgnoreWhenComplex<false, false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"eigenvalues", [](lua_State * L)
				{
					using ComplexType = Eigen::Matrix<std::complex<T::Scalar>, Eigen::Dynamic, Eigen::Dynamic>;
					
					bool bGetVectors = GetEigenvectors(L);	// mat, ignore?, "no_eigenvectors"
					auto td = GetTypeData<ComplexType>(L);

					luaL_argcheck(L, td, 1, "Complex matrix type unavailable for cast");

					Eigen::EigenSolver<T> es{*GetT(L), bGetVectors};

					ComplexType vals = es.eigenvalues(), vecs = es.eigenvectors();

					lua_getref(L, td->mPushRef);// mat, ignore?, "no_eigenvectors", push_new_type
					lua_pushlightuserdata(L, &vals);// mat, ignore?, "no_eigenvectors", push_new_type, eigenvalues_mat
					lua_call(L, 1, 1);	// mat, ignore?, "no_eigenvectors", conv_eigenvalues_mat

					if (bGetVectors)
					{
						lua_getref(L, td->mPushRef);// mat, ignore?, "no_eigenvectors", conv_eigenvalues_mat, push_new_type
						lua_pushlightuserdata(L, &vecs);// mat, ignore?, "no_eigenvectors", conv_eigenvalues_mat, push_new_type, eigenvectors_mat
						lua_call(L, 1, 1);	// mat, ignore?, "no_eigenvectors", conv_eigenvalues_mat, conv_eigenvectors_mat
					}

					return bGetVectors ? 2 : 1;
				}
			}, {
				"pseudoEigenvalues", [](lua_State * L)
				{					
					bool bGetVectors = GetEigenvectors(L);	// mat, ignore?, "no_eigenvectors"

					Eigen::EigenSolver<T> es{*GetT(L), bGetVectors};
					int ret = NewMoveRet<R>(L, es.pseudoEigenvalueMatrix());// mat, ignore?, "no_eigenvectors", values

					if (bGetVectors) ret += NewMoveRet<R>(L, es.pseudoEigenvectors());	// mat, ignore?, "no_eigenvectors", values, vecs

					return ret;
				}
			}, {
				"realEigenvalues", [](lua_State * L)
				{					
					bool bGetVectors = GetEigenvectors(L);	// mat, ignore?, "no_eigenvectors"

					Eigen::EigenSolver<T> es{*GetT(L), bGetVectors};

					int ret = NewMoveRet<T>(L, es.eigenvalues().real());// mat, ignore?, "no_eigenvectors", values

					if (bGetVectors) ret += NewMoveRet<T>(L, es.eigenvectors().real());// mat, ignore?, "no_eigenvectors", values, vecs

					return ret;
				}
			},
			{ nullptr, nullptr }
		};
		
		luaL_register(L, nullptr, methods);
	}

	//
	template<> void NoMap<false> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"__add", [](lua_State * L)
				{
					TwoMatrices<R> ms{L};

					return NewMoveRet<R>(L, *ms.mMat1 + *ms.mMat2);
				}
			}, {
				EIGEN_MATRIX_RESIZE_METHOD(conservativeResize)
			}, {
				EIGEN_MATRIX_PAIR_VOID_METHOD(conservativeResizeLike)
			}, {
				"replicate", [](lua_State * L)
				{
					int a = luaL_checkint(L, 2);

					if (lua_isstring(L, 3))
					{
						switch (GetReductionChoice(L, 3))
						{
						case eColwise:
							return NewMoveRet<R>(L, GetT(L)->colwise().replicate(a));
						case eRowwise:
							return NewMoveRet<R>(L, GetT(L)->rowwise().replicate(a));
						default:
							luaL_argcheck(L, false, 3, "Expected column rather than reduction choice");

							return 0;
						}
					}

					else return NewMoveRet<R>(L, GetT(L)->replicate(a, luaL_checkint(L, 3)));
				}
			}, 
		#ifdef WANT_MAP
			{
				"reshape", [](lua_State * L)
				{
					T & m = *GetT(L);
					Eigen::Map<T> map{m.data(), luaL_checkint(L, 2), luaL_checkint(L, 3)};

					return NewMoveRet<Eigen::Map<T>>(L, map);
				}
			}, {
				"reshapeWithInnerStride", [](lua_State * L)
				{
					T & m = *GetT(L);
					Eigen::Map<T, 0, Eigen::InnerStride<>> map{m.data(), luaL_checkint(L, 2), luaL_checkint(L, 3), luaL_checkint(L, 4)};

					return NewMoveRet<decltype(map)>(L, map);
				}
			}, {
				"reshapeWithOuterStride", [](lua_State * L)
				{
					T & m = *GetT(L);
					Eigen::Map<T, 0, Eigen::OuterStride<>> map{m.data(), luaL_checkint(L, 2), luaL_checkint(L, 3), luaL_checkint(L, 4)};

					return NewMoveRet<decltype(map)>(L, map);
				}
			},
		#endif
			{
				EIGEN_MATRIX_RESIZE_METHOD(resize)
			}, {
				EIGEN_MATRIX_PAIR_VOID_METHOD(resizeLike)
			}, {
				"__sub", [](lua_State * L)
				{
					TwoMatrices<R> ms{L};

					return NewMoveRet<R>(L, *ms.mMat1 - *ms.mMat2);
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		IgnoreWhenComplex(L);
	}
};