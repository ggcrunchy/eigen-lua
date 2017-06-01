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
#include <type_traits>
#include <Eigen/Eigen>

/***********************
* VectorwiseOp methods *
***********************/
template<typename U, int Dir, typename R> struct AttachMethods<Eigen::VectorwiseOp<U, Dir>, R> : InstanceGetters<Eigen::VectorwiseOp<U, Dir>, R> {
	using VR = VectorRef<R, Eigen::VectorwiseOp<U, Dir>::isHorizontal>;

	// Add some methods when the underlying type is real.
	template<bool = !Eigen::NumTraits<R::Scalar>::IsComplex> void NonComplex (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(maxCoeff)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(minCoeff)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	template<> void NonComplex<false> (lua_State *) {}

	// Add methods that only makes sense for non-boolean matrices.
	template<typename M = R> void AddBoolMatrixDependent (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"add", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L) + *VR{L, 2});
				}
			}, {
				"addInPlace", [](lua_State * L)
				{
					*GetT(L) += *VR{L, 2};

					return SelfForChaining(L);
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(blueNorm)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(hypotNorm)
			}, {
				"lp1Norm", [](lua_State * L)
				{
					EIGEN_MATRIX_GET_MATRIX(lpNorm<1>);
				}
			}, {
				"lpInfNorm", [](lua_State * L)
				{
					EIGEN_MATRIX_GET_MATRIX(lpNorm<Eigen::Infinity>);
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(mean)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(norm)
			}, {
				EIGEN_MATRIX_VOID_METHOD(normalize)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(normalized)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(prod)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(squaredNorm)
			}, {
				"sub", [](lua_State * L)
				{
					return NewRet<R>(L, *GetT(L) - *VR{L, 2});
				}
			}, {
				"subInPlace", [](lua_State * L)
				{
					*GetT(L) -= *VR(L, 2);

					return SelfForChaining(L);
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(sum)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		NonComplex(L);
	}

	// Add methods for BoolMatrix-based types.
	template<> void AddBoolMatrixDependent<BoolMatrix> (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_GET_MATRIX_METHOD(all)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(any)
			}, {
				"count", [](lua_State * L)
				{
					auto td = TypeData<Eigen::MatrixXi>::Get(L, GetTypeData::eFetchIfMissing);

					luaL_argcheck(L, td, 2, "Vectorwise count() requires int matrices");

					Eigen::MatrixXi im = GetT(L)->count();

					PUSH_TYPED_DATA(im);
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	AttachMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"assign", [](lua_State * L)
				{
					*GetT(L) = *VR{L, 2};

					return SelfForChaining(L);
				}
			}, {
				"redux", [](lua_State * L)
				{
					return NewRet<R>(L, Redux<Eigen::VectorwiseOp<U, Dir>, R, R>(L));
				}
			}, {
				EIGEN_MATRIX_GET_MATRIX_COUNT_METHOD(replicate)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(reverse)
			}, {
				EIGEN_MATRIX_VOID_METHOD(reverseInPlace)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		AddBoolMatrixDependent(L);
	}
};

template<typename U, int Dir> struct AuxTypeName<Eigen::VectorwiseOp<U, Dir>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "VectorwiseOp<");

		AuxTypeName<U>(B, L);

		lua_pushfstring(L, ", %d>", Dir);	// ..., Dir
		luaL_addvalue(B);	// ...
	}
};