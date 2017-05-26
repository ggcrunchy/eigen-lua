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
#include "ByteReader.h"
#include "types.h"
#include "utils.h"
#include "macros.h"
#include <algorithm>
#include <type_traits>

#define COEFF_MUTATE(OP)	T & m = *GetT(L);												\
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

template<typename R, typename T> struct MutateOp {
	lua_State * mL;
	R mTemp;

	MutateOp (lua_State * L) : mL{L}
	{
	}

	#define VB_OP(OP)	void operator OP (bool)																											\
						{																																\
							T & object = *GetInstance<T>(mL);																							\
																																						\
							if (HasType<T>(mL, 2)) object OP *LuaXS::UD<T>(mL, 2);																		\
							else if (HasType<R>(mL, 2)) object OP *LuaXS::UD<R>(mL, 2);																	\
							else if (HasType<Eigen::Block<R>>(mL, 2)) object OP *LuaXS::UD<Eigen::Block<R>>(mL, 2);										\
							else if (HasType<Eigen::Transpose<R>>(mL, 2)) object OP *LuaXS::UD<Eigen::Transpose<R>>(mL, 2);								\
							else if (HasType<Eigen::Block<Eigen::Transpose<R>>>(mL, 2)) object OP *LuaXS::UD<Eigen::Block<Eigen::Transpose<R>>>(mL, 2);	\
							else object OP *SetTemp(mL, &mTemp, 2);																						\
						}

	VB_OP(=)
	VB_OP(+=)
	VB_OP(/=)
	VB_OP(*=)
	VB_OP(-=)

	#undef VB_OP
};

template<typename R, typename U> struct MutateOp<R, Eigen::Diagonal<U, Eigen::DynamicIndex>> {
	lua_State * mL;
	R mTemp;

	MutateOp (lua_State * L) : mL{L}
	{
	}

	#define VB_OP(OP)	void operator OP (bool)																																							\
						{																																												\
							auto & object = *GetInstance<Eigen::Diagonal<U, Eigen::DynamicIndex>>(mL, 1);																								\
																																																		\
							if (HasType<Eigen::Diagonal<U, Eigen::DynamicIndex>>(mL, 2)) object OP *GetInstance<Eigen::Diagonal<U, Eigen::DynamicIndex>>(mL, 2);										\
							else if (HasType<Eigen::VectorBlock<U, Eigen::Dynamic>>(mL, 2)) object OP *GetInstance<Eigen::VectorBlock<U, Eigen::Dynamic>>(mL, 2);										\
							else if (HasType<Eigen::VectorBlock<Eigen::Transpose<U>, Eigen::Dynamic>>(mL, 2)) object OP *GetInstance<Eigen::VectorBlock<Eigen::Transpose<U>, Eigen::Dynamic>>(mL, 2);	\
							else object OP AsVector<R>::To(HasType<R>(mL, 2) ? LuaXS::UD<R>(mL, 2) : SetTemp(mL, &mTemp, 2));																			\
						}

	VB_OP(=)
	VB_OP(+=)
	VB_OP(/=)
	VB_OP(*=)
	VB_OP(-=)

	#undef VB_OP
};

template<typename R, typename U, int N> struct MutateOp<R, Eigen::VectorBlock<U, N>> {
	lua_State * mL;
	R mTemp;

	MutateOp (lua_State * L) : mL{L}
	{
	}

	#define VB_OP(OP)	void operator OP (bool)																						\
						{																											\
							auto & object = *GetInstance<Eigen::VectorBlock<U, N>>(mL, 1);											\
																																	\
							if (HasType<Eigen::VectorBlock<U, N>>(mL, 2)) object OP *GetInstance<Eigen::VectorBlock<U, N>>(mL, 2);	\
							else object OP AsVector<R>::To(HasType<R>(mL, 2) ? LuaXS::UD<R>(mL, 2) : SetTemp(mL, &mTemp, 2));		\
						}

	VB_OP(=)
	VB_OP(+=)
	VB_OP(/=)
	VB_OP(*=)
	VB_OP(-=)

	#undef VB_OP
};

//
#define MUTATE(OP)	auto how = GetVectorwiseOption(L, 3);									\
																							\
					if (how == eNotVectorwise)												\
					{																		\
						MutateOp<R, T> mo{L};												\
																							\
						mo OP true;															\
					}																		\
																							\
					else																	\
					{																		\
						if (how == eColwise) GetT(L)->colwise() OP AsVector<R>::To(L, 2);	\
						else GetT(L)->rowwise() OP AsVector<R>::To(L, 2).transpose();		\
					}																		\
																							\
					return SelfForChaining(L)

//
#define IN_PLACE_REDUCE(METHOD)	auto how = GetVectorwiseOption(L, 3);					\
																						\
								if (how == eNotVectorwise) GetT(L)->METHOD();			\
																						\
								else													\
								{														\
									if (how == eColwise) GetT(L)->colwise().METHOD();	\
									else GetT(L)->rowwise().METHOD();					\
								}														\
																						\
								return 0

#define COEFF_MUTATE_METHOD(NAME, OP) EIGEN_REG(NAME, COEFF_MUTATE(OP))
#define IN_PLACE_REDUCE_METHOD(NAME) EIGEN_REG(NAME, IN_PLACE_REDUCE(NAME))

//
template<typename T, typename R = T> struct WriteOps {
	ADD_INSTANCE_GETTERS()

	template<bool = IsXpr<T>::value> void AddNonXpr (lua_State * L) // todo: can probably be more precise, e.g. blocks of maps with inner or outer stride
	{
		luaL_Reg methods[] = {
			{
				IN_PLACE_REDUCE_METHOD(reverseInPlace)
			}, {
				EIGEN_MATRIX_VOID_METHOD(transposeInPlace)
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}

	template<> void AddNonXpr<true> (lua_State *) {}

	template<bool = !std::is_same<R, BoolMatrix>::value> void AddNonBool (lua_State * L)
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
				COEFF_MUTATE_METHOD(coeffAssign, =)
			}, {
				COEFF_MUTATE_METHOD(coeffDivInPlace, /=)
			}, {
				COEFF_MUTATE_METHOD(coeffMulInPlace, *=)
			}, {
				COEFF_MUTATE_METHOD(coeffSubInPlace, -=)
			}, {
				IN_PLACE_REDUCE_METHOD(normalize)
			}, {
				"setFromBytes", [](lua_State * L)
				{
					ByteReader bytes{L, 2};
					T & m = *GetT(L);

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
				EIGEN_MATRIX_CHAIN_METHOD(setIdentity)
			}, {
				"setLinSpaced", [](lua_State * L)
				{
					T & m = *GetT(L);

					CheckVector(L, m, 1);

					if (m.cols() == 1) m = LinSpacing<T, Eigen::Dynamic, 1>::Make(L, m.rows());
					else m = LinSpacing<T, 1, Eigen::Dynamic>::Make(L, m.cols());

					return SelfForChaining(L);
				}
			}, {
				"stableNormalize", [](lua_State * L)
				{
					AsVector<R>::To(L).stableNormalize();

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

	template<> void AddNonBool<false> (lua_State * L) {}

	WriteOps (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"assign", [](lua_State * L)
				{
					MUTATE(=);
				}
			}, {
				EIGEN_MATRIX_SET_SCALAR_METHOD(fill)
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
					T & object = *GetT(L);

					if (HasType<T>(L, 2)) object.swap(*LuaXS::UD<T>(L, 2));
					else if (HasType<R>(L, 2)) object.swap(*LuaXS::UD<R>(L, 2));
					else if (HasType<Eigen::Block<R>>(L, 2)) object.swap(*LuaXS::UD<Eigen::Block<R>>(L, 2));
					else if (HasType<Eigen::Transpose<R>>(L, 2)) object.swap(*LuaXS::UD<Eigen::Transpose<R>>(L, 2));
					else if (HasType<Eigen::Block<Eigen::Transpose<R>>>(L, 2)) object.swap(*LuaXS::UD<Eigen::Block<Eigen::Transpose<R>>>(L, 2));
					else
					{
						R t1 = object, t2;

						SetTemp(L, &t2, 2);

						luaL_argcheck(L, luaL_getmetafield(L, 2, "assign"), 2, "Type has no assign method");// object, other, assign
						lua_insert(L, 2);	// object, assign, other

						New<R>(L, t1);	// object, assign, other, object_mat

						lua_call(L, 2, 0);	// object

						object = t2;
					}

					return 0;
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		AddNonBool(L);
		AddNonXpr(L);
	}
};