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

#include "types.h"
#include "utils.h"
#include "macros.h"

//
template<typename T, typename R, bool = IsBasic<T>::value> struct WriteOpsGetters : InstanceGetters<T, R> {};
template<typename T, typename R> struct WriteOpsGetters<T, R, false> : TempInstanceGetters<T, R> {};

//
template<typename T, typename R, bool = IsLvalue<T>::value> struct WriteOps : WriteOpsGetters<T, R> {
	//
	template<typename = T> struct MutateOp {
		lua_State * mL;
		R mTemp;

		MutateOp (lua_State * L) : mL{L}
		{
		}

		#define VB_OP(OP)	void operator OP (bool)																				\
							{																									\
								auto t = GetT(mL);																				\
								auto & object = *t;																				\
																																\
								if (HasType<T>(mL, 2)) object OP *LuaXS::UD<T>(mL, 2);											\
								else if (HasType<Eigen::Block<R>>(mL, 2)) object OP *LuaXS::UD<Eigen::Block<R>>(mL, 2);			\
								else if (HasType<Eigen::Transpose<R>>(mL, 2)) object OP *LuaXS::UD<Eigen::Transpose<R>>(mL, 2);	\
								else object OP *SetTemp(mL, &mTemp, 2);															\
							}

		VB_OP(=)
		VB_OP(+=)
		VB_OP(/=)
		VB_OP(*=)
		VB_OP(-=)

		#undef VB_OP
	};

	template<typename U, int I> struct MutateOp<Eigen::Diagonal<U, I>> {
		lua_State * mL;

		MutateOp (lua_State * L) : mL{L}
		{
		}

		#define VB_OP(OP)	void operator OP (bool)																														\
							{																																			\
								auto t = GetT(mL, 1);																													\
								auto & object = *t;																														\
																																										\
								if (HasType<Eigen::Diagonal<U, I>>(mL, 2)) object OP *GetInstance<Eigen::Diagonal<U, I>>(mL, 2);										\
								else if (HasType<Eigen::VectorBlock<U, Eigen::Dynamic>>(mL, 2)) object OP *GetInstance<Eigen::VectorBlock<U, Eigen::Dynamic>>(mL, 2);	\
								else object OP *ColumnVector<R>(mL, 2);																									\
							}

		VB_OP(=)
		VB_OP(+=)
		VB_OP(/=)
		VB_OP(*=)
		VB_OP(-=)

		#undef VB_OP
	};

	template<typename U, int Size> struct MutateOp<Eigen::VectorBlock<U, Size>> {
		lua_State * mL;

		MutateOp (lua_State * L) : mL{L}
		{
		}

		#define VB_OP(OP)	void operator OP (bool)																							\
							{																												\
								auto t = GetT(mL, 1);																						\
								auto & object = *t;																							\
																																			\
								if (HasType<Eigen::VectorBlock<U, Size>>(mL, 2)) object OP *GetInstance<Eigen::VectorBlock<U, Size>>(mL, 2);\
								else object OP *ColumnVector<R>(mL, 2);																		\
							}	// TODO: ^^^^ or row vector...

		VB_OP(=)
		VB_OP(+=)
		VB_OP(/=)
		VB_OP(*=)
		VB_OP(-=)

		#undef VB_OP
	};

	// State usable by any of the resize methods' overloads.
	struct ResizeState {
		int mDim1{1}, mDim2{1};
		bool mHas1{true}, mHas2{true};

		ResizeState (lua_State * L, const T & mat)
		{
			if (!lua_isnoneornil(L, 2))
			{
				lua_pushliteral(L, "NoChange");	// mat, a, b, "NoChange"

				mHas1 = lua_equal(L, 2, -1) != 0;
				mHas2 = lua_equal(L, 3, -1) != 0;

				luaL_argcheck(L, mHas1 || mHas2, 1, "Must resize at least one dimension");

				if (mHas1) mDim1 = LuaXS::Int(L, 2);
				if (mHas2) mDim2 = LuaXS::Int(L, 3);
			}

			else
			{
				CheckVector(L, mat, 1);

				int a = LuaXS::Int(L, 2);

				if (mat.cols() == 1) mDim2 = a;

				else mDim1 = a;
			}
		}
	};

	//
	#define COEFF_MUTATE(OP)	auto t = GetT(L);												\
								auto & m = *t;													\
								int a = LuaXS::Int(L, 2) - 1;									\
																								\
								if (lua_gettop(L) == 3)											\
								{																\
									CheckVector(L, m, 1);										\
																								\
									(m.cols() == 1 ? m(a, 0) : m(0, a)) OP AsScalar<R>(L, 3);	\
								}																\
																								\
								else m(a, LuaXS::Int(L, 3) - 1) OP AsScalar<R>(L, 4);			\
																								\
								return 0

	//
	#define MUTATE(OP)	MutateOp<> mo{L};			\
													\
						mo OP true;					\
													\
						return SelfForChaining(L)

	// Common form of resize methods.
	#define EIGEN_MATRIX_RESIZE(METHOD)	auto t = GetT(L);										\
										auto & m = *t;											\
										ResizeState rs{L, m};									\
																								\
										if (!rs.mHas2) m.METHOD(rs.mDim1, Eigen::NoChange);		\
										else if (!rs.mHas1) m.METHOD(Eigen::NoChange, rs.mDim2);\
										else m.METHOD(rs.mDim1, rs.mDim2);						\
																								\
										return 0

	//
	#define COEFF_MUTATE_METHOD(NAME, OP) EIGEN_REG(NAME, COEFF_MUTATE(OP))
	#define EIGEN_MATRIX_RESIZE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_RESIZE(NAME))

	// Operations added for matrices.
	template<bool = IsMatrix<T>::value> void AddMatrix (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_RESIZE_METHOD(conservativeResize)
			}, {
				EIGEN_MATRIX_PAIR_VOID_METHOD(conservativeResizeLike)
			}, {
				EIGEN_MATRIX_RESIZE_METHOD(resize)
			}, {
				EIGEN_MATRIX_PAIR_VOID_METHOD(resizeLike)
			}, {
				EIGEN_MATRIX_VOID_METHOD(transposeInPlace)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	};

	//
	template<bool> void AddTransposeInPlace (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"transposeInPlace", [](lua_State * L)
				{
					TempRAII<T> object{L};

					luaL_argcheck(L, object->cols() == object->rows(), 1, "Attempt to transpose non-square, non-matrix object in place");

					object->transposeInPlace();

					return 0;
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	template<> void AddTransposeInPlace<false> (lua_State *) {}

	// No-op when not a raw matrix.
	template<> void AddMatrix<false> (lua_State * L)
	{
		using N = TempRAII<T>::N;

		AddTransposeInPlace<N::RowsAtCompileTime == N::ColsAtCompileTime>(L);
	}

	//
	template<typename = R> void AddNonBool (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_VOID_METHOD(adjointInPlace)
			}, {
				"addInPlace", [](lua_State * L)
				{
					MUTATE(+=);
				}
			}, {
				COEFF_MUTATE_METHOD(coeffAddInPlace, +=)
			}, {
				COEFF_MUTATE_METHOD(coeffDivInPlace, /=)
			}, {
				COEFF_MUTATE_METHOD(coeffMulInPlace, *=)
			}, {
				COEFF_MUTATE_METHOD(coeffSubInPlace, -=)
			}, {
				EIGEN_MATRIX_VOID_METHOD(normalize)
			}, {
				"setFromBytes", [](lua_State * L)
				{
					ByteReader bytes{L, 2};
					auto t = GetT(L);
					auto & m = *t;

					if (!bytes.mBytes) lua_error(L);

					size_t row = 0, col = 0, n = (std::min)(bytes.mCount / sizeof(R::Scalar), size_t(m.size()));

					for (size_t i = 0; i < n; ++i)
					{
						m.coeffRef(row, col) = reinterpret_cast<const R::Scalar *>(bytes.mBytes)[i];

						if (++col == m.cols())
						{
							col = 0;

							++row;
						}
					}

					return SelfForChaining(L);
				}
			}, {
				"setLinSpaced", [](lua_State * L)
				{
					auto t = GetT(L);
					auto & m = *t;

					CheckVector(L, m, 1);

					if (m.cols() == 1) m = LinSpacing<T, Eigen::Dynamic, 1>::Make(L, m.rows());
					else m = LinSpacing<T, 1, Eigen::Dynamic>::Make(L, m.cols());

					return SelfForChaining(L);
				}
			}, {
				"stableNormalize", [](lua_State * L)
				{
					ColumnVector<R> cv{L, 1};

					cv->stableNormalize();

					cv.RestoreShape();

					if (cv.Changed()) *GetT(L) = *cv;

					return 0;
				}
			}, {
				"subInPlace", [](lua_State * L)
				{
					MUTATE(-=);
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	//
	template<> void AddNonBool<BoolMatrix> (lua_State * L) {}

	WriteOps (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"assign", [](lua_State * L)
				{
					MUTATE(=);
				}
			}, {
				COEFF_MUTATE_METHOD(coeffAssign, =)
			}, {
				EIGEN_MATRIX_SET_SCALAR_METHOD(fill)
			}, {
				EIGEN_MATRIX_VOID_METHOD(reverseInPlace)
			}, {
				EIGEN_MATRIX_SET_SCALAR_CHAIN_METHOD(setConstant)
			}, {
				EIGEN_MATRIX_CHAIN_METHOD(setIdentity)
			}, {
				EIGEN_MATRIX_CHAIN_METHOD(setOnes)
			}, {
				EIGEN_MATRIX_CHAIN_METHOD(setRandom)
			}, {
				EIGEN_MATRIX_CHAIN_METHOD(setZero)
			}, {
				"swap", [](lua_State * L)
				{
					auto t = GetT(L);
					auto & m = *t;

					if (HasType<T>(L, 2)) m.swap(*LuaXS::UD<T>(L, 2));
					else if (HasType<R>(L, 2)) m.swap(*LuaXS::UD<R>(L, 2)); // N.B. preempts same logic in SetTemp()
					else if (HasType<Eigen::Block<R>>(L, 2)) m.swap(*LuaXS::UD<Eigen::Block<R>>(L, 2));
					else if (HasType<Eigen::Transpose<R>>(L, 2)) m.swap(*LuaXS::UD<Eigen::Transpose<R>>(L, 2));
					else
					{
						R t1 = m, t2, * pt2 = SetTemp(L, &t2, 2);

						luaL_argcheck(L, luaL_getmetafield(L, 2, "assign"), 2, "Type has no assign method");// object, other, assign
						lua_insert(L, 2);	// m, assign, other

						New<R>(L, t1);	// m, assign, other, m_mat

						lua_call(L, 2, 0);	// m

						m = *pt2;
					}

					return 0;
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		AddMatrix(L);
		AddNonBool(L);
	}
};

// Leave write ops out if the object is not an lvalue.
template<typename T, typename R> struct WriteOps<T, R, false> {
	WriteOps (lua_State *) {}
};