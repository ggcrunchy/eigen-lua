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

//
EXTENDED_UNRAVELER(Eigen::SelfAdjointView);

//
enum {
	eSelfAdjointViewIDs = eSelfAdjointViewMarker,

	VIEW_ENUMS(eSAV_L),	// Lower-triangular self-adjoint view
	VIEW_ENUMS(eSAV_U)	// Lower-triangular self-adjoint view
};

#define ADD_COMPLEX_TYPES(VIEW, ENUM, EX, T, NAME)

//
#define UNROLL_SAV_UNRAVELERS(T)	UNROLL_VIEW(Eigen::SelfAdjointView, eSAV_L, Eigen::Lower, T, "_savl");	\
									UNROLL_VIEW(Eigen::SelfAdjointView, eSAV_U, Eigen::Upper, T, "_savu")

//
#ifdef WANT_FLOAT
	UNROLL_SAV_UNRAVELERS(Eigen::MatrixXf);
#endif

#ifdef WANT_DOUBLE
	UNROLL_SAV_UNRAVELERS(Eigen::MatrixXd);
#endif


#if defined(WANT_CFLOAT) || defined(WANT_CDOUBLE)
	#undef ADD_COMPLEX_TYPES
	#define ADD_COMPLEX_TYPES ADD_COMPLEX_VIEW_TYPES
#endif

//
#ifdef WANT_CFLOAT
	UNROLL_SAV_UNRAVELERS(Eigen::MatrixXcf);
#endif

#ifdef WANT_CDOUBLE
	UNROLL_SAV_UNRAVELERS(Eigen::MatrixXcd);
#endif

#undef ADD_COMPLEX_TYPES

//
template<typename T, typename R> struct AttachSelfAdjointViewMethods {
	ADD_INSTANCE_GETTERS()

	//
	template<int> void DoMethod (lua_State * L) {}

	//
	template<bool = true> static void BaseDerivedSAV(lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"assign", [](lua_State * L)
				{
					return 0;	// TODO
				}
			}, {
				"coeffAssign", [](lua_State * L)
				{
					return 0;	// TODO
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(innerStride)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(outerStride)
			}, {
				"rankUpdate", [](lua_State * L)
				{
					auto & v = *GetT(L);
					bool bHasV = LuaXS::IsType(L, FullName<R>(), 3);
					int spos = bHasV ? 4 : 3;
					R::Scalar alpha = !lua_isnoneornil(L, spos) ? ArgObject<R>{}.AsScalar(L, spos) : R::Scalar(1);

					if (bHasV) v.rankUpdate(AsVector<R>::To(L, 2), AsVector<R>::To(L, 3), alpha);
					else v.rankUpdate(AsVector<R>::To(L, 2), alpha);

					return SelfForChaining(L);	// v, u[, v][, scalar], v
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	template<typename U> struct DerivedSAV {
		//
		template<typename V> struct NotMap {
			NotMap (lua_State * L)
			{
				luaL_Reg methods[] = {
					{
						"triangularView", [](lua_State * L)
						{
							auto & v = *GetT(L);
							const char * names[] = { "lower", "strictly_lower", "strictly_upper", "unit_lower", "unit_upper", "upper", nullptr };

							#define PUSH_RES(METHOD) { EIGEN_PUSH_AUTO_RESULT(METHOD); }

							switch (luaL_checkoption(L, 2, nullptr, names))
							{
							case 0:	// Lower-triangular
								PUSH_RES(triangularView<Eigen::Lower>)
							case 1:// Strictly lower-triangular
								PUSH_RES(triangularView<Eigen::StrictlyLower>)
							case 2:// Strictly upper-triangular
								PUSH_RES(triangularView<Eigen::StrictlyUpper>)
							case 3:// Upper-triangular
								PUSH_RES(triangularView<Eigen::UnitLower>)
							case 4:// Unit lower-triangular
								PUSH_RES(triangularView<Eigen::UnitUpper>)
							default:// Unit upper-triangular
								PUSH_RES(triangularView<Eigen::Upper>)
							}

							#undef PUSH_RES
						}
					},
					{ nullptr, nullptr }
				};

				luaL_register(L, nullptr, methods);
			};
		};

		template<typename V, int O, typename S, unsigned int UL> struct NotMap<Eigen::SelfAdjointView<Eigen::Map<V, O, S>, UL>> {
			NotMap (lua_State *) {}
		};

		DerivedSAV(lua_State * L)
		{
			luaL_Reg methods[] = {
				{
					EIGEN_PUSH_AUTO_RESULT_METHOD(adjoint)
				}, {
					EIGEN_PUSH_AUTO_RESULT_METHOD(conjugate)
				}, {
					EIGEN_PUSH_AUTO_RESULT_METHOD(ldlt) // n.b. earlier LDLT<T, UpLo> for normal matrices, i.e. not row-major, thus here
				}, {
					EIGEN_PUSH_AUTO_RESULT_METHOD(llt)	// ditto
				}, {
					EIGEN_PUSH_AUTO_RESULT_METHOD(transpose)
				},
				{ nullptr, nullptr }
			};

			luaL_register(L, nullptr, methods);

			BaseDerivedSAV(L);

			NotMap<U> nm{L};
		}
	};

	//
	template<typename U, unsigned int N> struct DerivedSAV<Eigen::SelfAdjointView<Eigen::Transpose<U>, N>> {
		DerivedSAV (lua_State * L)
		{
			BaseDerivedSAV(L);
		}
	};

	//
	template<typename U, unsigned int N> struct DerivedSAV<Eigen::SelfAdjointView<const U, N>> {
		DerivedSAV (lua_State *) {}
	};

	template<typename U, unsigned int N> struct DerivedSAV<const Eigen::SelfAdjointView<const U, N>> {
		DerivedSAV (lua_State *) {}
	};

	template<typename U, typename V, unsigned int N> struct DerivedSAV<Eigen::SelfAdjointView<Eigen::CwiseUnaryOp<U, V>, N>> {
		DerivedSAV (lua_State *) {}
	};

	//
	template<bool = Eigen::NumTraits<R::Scalar>::IsComplex> static int EigenvaluesSAV (lua_State * L)
	{
		EIGEN_MATRIX_GET_MATRIX(eigenvalues);
	}

	template<> static int EigenvaluesSAV<false> (lua_State * L)
	{
		USING_COMPLEX_TYPE();
		EIGEN_REAL_GET_COMPLEX(eigenvalues);
	}

	/************************************
	* Lower-triangular self-adjoint view
	************************************/
	template<> void DoMethod<eSAV_L>(lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"asMatrix", [](lua_State * L)
				{
					R matrix = *GetT(L);

					return NewMoveRet<R>(L, matrix);
				}
			}, {
				"__call", [](lua_State * L)
				{
					return 1;	// TODO
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(cols)
			}, {
				EIGEN_MATRIX_GET_MATRIX_METHOD(diagonal)
			}, {
				"eigenvalues", EigenvaluesSAV<>
			}, {
				"__mul", [](lua_State * L)
				{
					/*
					template<typename OtherDerived>
					117     EIGEN_DEVICE_FUNC
					118     const Product<SelfAdjointView,OtherDerived>
					119     operator*(const MatrixBase<OtherDerived>& rhs) const
					120     {
					121       return Product<SelfAdjointView,OtherDerived>(*this, rhs.derived());
					122     }
					123
					125     template<typename OtherDerived> friend
					126     EIGEN_DEVICE_FUNC
					127     const Product<OtherDerived,SelfAdjointView>
					128     operator*(const MatrixBase<OtherDerived>& lhs, const SelfAdjointView& rhs)
					129     {
					130       return Product<OtherDerived,SelfAdjointView>(lhs.derived(),rhs);
					131     }
					132
					133     friend EIGEN_DEVICE_FUNC
					134     const SelfAdjointView<const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(Scalar,MatrixType,product),UpLo>
					135     operator*(const Scalar& s, const SelfAdjointView& mat)
					136     {
					137       return (s*mat.nestedExpression()).template selfadjointView<UpLo>();
					138     }
					*/
					return 1;	// TODO
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(operatorNorm)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rows)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		DerivedSAV<T> derived{L};
	}

	VIEW_METHODS(eSAV_L)

	/************************************
	* Upper-triangular self-adjoint view
	************************************/
	template<> void DoMethod<eSAV_U> (lua_State * L)
	{
		DoMethod<eSAV_L>(L);
	}

	VIEW_METHODS(eSAV_U)

	//
	AttachSelfAdjointViewMethods (lua_State * L)
	{
		DoMethod<ObjectTypeID<T>::kID>(L);
	}
};

