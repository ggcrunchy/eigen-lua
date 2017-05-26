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

#include "views.h"

/*************************
* TriangularView methods *
*************************/
template<typename MT, unsigned int UpLo, typename R> struct AttachMethods<Eigen::TriangularView<MT, UpLo>, R> {
	using T = Eigen::TriangularView<MT, UpLo>;

	ADD_INSTANCE_GETTERS()

	//
	template<int ModeAnded> static int GetSelfAdjointView (lua_State * L)
	{
		return luaL_error(L, "Only upper or lower triangular views may yield self-adjoint views");
	}

	template<> static int GetSelfAdjointView<0> (lua_State * L)
	{
		EIGEN_PUSH_AUTO_RESULT(selfadjointView);
	}

	//
	template<typename U> struct DerivedTV {
		DerivedTV (lua_State * L)
		{
			luaL_Reg methods[] = {
				{
					EIGEN_PUSH_AUTO_RESULT_METHOD(adjoint)
				}, {
					EIGEN_PUSH_AUTO_RESULT_METHOD(conjugate)
				}, {
					"selfadjointView", [](lua_State * L)
					{
						return GetSelfAdjointView<T::Mode & (Eigen::Lower | Eigen::Upper)>(L);
					}
				}, {
					EIGEN_PUSH_AUTO_RESULT_METHOD(transpose)
				},
				{ nullptr, nullptr }
			};

			luaL_register(L, nullptr, methods);
		}
	};

	//
	template<typename U, unsigned int N> struct DerivedTV<Eigen::TriangularView<Eigen::Transpose<U>, N>> {
		DerivedTV (lua_State * L)
		{
			luaL_Reg methods[] = {
				{
					EIGEN_MATRIX_SET_SCALAR_METHOD(fill)
				}, {
					EIGEN_MATRIX_SET_SCALAR_CHAIN_METHOD(setConstant)
				}, {
					EIGEN_MATRIX_CHAIN_METHOD(setOnes)
				}, {
					EIGEN_MATRIX_CHAIN_METHOD(setZero)
				},
				{ nullptr, nullptr }
			};

			luaL_register(L, nullptr, methods);
			//"assign" (matrix or another TV...)
			/*
			367     template<typename Other>
			368     EIGEN_DEVICE_FUNC
			369     TriangularViewType&  operator+=(const DenseBase<Other>& other) {
			370       internal::call_assignment_no_alias(derived(), other.derived(), internal::add_assign_op<Scalar,typename Other::Scalar>());
			371       return derived();
			372     }
			374     template<typename Other>
			375     EIGEN_DEVICE_FUNC
			376     TriangularViewType&  operator-=(const DenseBase<Other>& other) {
			377       internal::call_assignment_no_alias(derived(), other.derived(), internal::sub_assign_op<Scalar,typename Other::Scalar>());
			378       return derived();
			379     }
			380
			382     EIGEN_DEVICE_FUNC
			383     TriangularViewType&  operator*=(const typename internal::traits<MatrixType>::Scalar& other) { return *this = derived().nestedExpression() * other; }
			385     EIGEN_DEVICE_FUNC
			386     TriangularViewType&  operator/=(const typename internal::traits<MatrixType>::Scalar& other) { return *this = derived().nestedExpression() / other; }

			424     template<typename OtherDerived>
			425     EIGEN_DEVICE_FUNC
			426     TriangularViewType& operator=(const TriangularBase<OtherDerived>& other);
			427
			429     template<typename OtherDerived>
			430     EIGEN_DEVICE_FUNC
			431     TriangularViewType& operator=(const MatrixBase<OtherDerived>& other);
			448
			514     template<typename OtherDerived>
			515     EIGEN_DEVICE_FUNC
			516 #ifdef EIGEN_PARSED_BY_DOXYGEN
			517     void swap(TriangularBase<OtherDerived> &other)
			518 #else
			519     void swap(TriangularBase<OtherDerived> const & other)
			520 #endif
			521     {
			522       EIGEN_STATIC_ASSERT_LVALUE(OtherDerived);
			523       call_assignment(derived(), other.const_cast_derived(), internal::swap_assign_op<Scalar>());
			524     }
			525
			528     template<typename OtherDerived>
			529     EIGEN_DEVICE_FUNC
			530     void swap(MatrixBase<OtherDerived> const & other)
			531     {
			532       EIGEN_STATIC_ASSERT_LVALUE(OtherDerived);
			533       call_assignment(derived(), other.const_cast_derived(), internal::swap_assign_op<Scalar>());
			534     }
			*/
		}
	};

	template<typename U, unsigned int N> struct DerivedTV<Eigen::TriangularView<const U, N>> {
		DerivedTV (lua_State *) {}
	};

	template<typename U, unsigned int N> struct DerivedTV<const Eigen::TriangularView<const U, N>> {
		DerivedTV (lua_State *) {}
	};

	template<typename U, typename V, unsigned int N> struct DerivedTV<Eigen::TriangularView<Eigen::CwiseUnaryOp<U, V>, N>> {
		DerivedTV (lua_State *) {}
	};

	//
	AttachMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"asMatrix", AsMatrix<T, R>
			}, {
				"__call", [](lua_State * L)
				{
					return 0;	// TODO
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(cols)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(determinant)
			}, /*{
			   EIGEN_MATRIX_PUSH_VALUE_METHOD(innerStride)
			}, */{
				"__gc", LuaXS::TypedGC<T>
			}, {
				"__mul", [](lua_State * L)
				{
					return 1;	// TODO
				}
			}, /*{
			   EIGEN_MATRIX_PUSH_VALUE_METHOD(outerStride)
			}, */{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rows)
			}, {
				"solve", [](lua_State * L)
				{
					if (WantsBool(L, "on_the_right")) return NewRet<R>(L, GetT(L)->solve<Eigen::OnTheRight>(GetR(L, 2)));
					else return NewRet<R>(L, GetT(L)->solve(GetR(L, 2)));
				}
			}, {
				"solveInPlace", [](lua_State * L) // todo: stumped here with TV<Transpose<Map<Matrix, 0, InnerStride>>,9>
				{
					if (WantsBool(L, "on_the_right")) GetT(L)->solveInPlace<Eigen::OnTheRight>(GetR(L, 2));
					else GetT(L)->solveInPlace(GetR(L, 2));

					return 0;
				}
			},
			{ nullptr, nullptr }
		};
		/*
		450     template<typename OtherDerived>
		451     EIGEN_DEVICE_FUNC
		452     const Product<TriangularViewType,OtherDerived>
		453     operator*(const MatrixBase<OtherDerived>& rhs) const
		454     {
		455       return Product<TriangularViewType,OtherDerived>(derived(), rhs.derived());
		456     }
		457
		459     template<typename OtherDerived> friend
		460     EIGEN_DEVICE_FUNC
		461     const Product<OtherDerived,TriangularViewType>
		462     operator*(const MatrixBase<OtherDerived>& lhs, const TriangularViewImpl& rhs)
		463     {
		464       return Product<OtherDerived,TriangularViewType>(lhs.derived(),rhs.derived());
		465     }
		466

		*/
		luaL_register(L, nullptr, methods);

		DerivedTV<T> derived{L};
	}
};

template<typename U, unsigned int E> struct AuxTypeName<Eigen::TriangularView<U, E>> {
	AuxTypeName (luaL_Buffer * B, lua_State * L)
	{
		luaL_addstring(B, "TriangularView<");

		AuxTypeName<U>(B, L);

		lua_pushfstring(L, ", %d>", E);	// ..., E
		luaL_addvalue(B);	// ...
	}
};