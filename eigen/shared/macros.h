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
#include "bool_matrix.h"

//
#define EIGEN_REL_OP(OP)	auto td = GetTypeData<BoolMatrix>(L);							\
																							\
							luaL_argcheck(L, td, 1, "BoolMatrix type unavailable");			\
																							\
							ArgObject<R> ao{L, 2};											\
							BoolMatrix bm;													\
																							\
							if (ao.mMat) bm = GetT(L)->array() OP ao.mMat->array();			\
																							\
							else bm = GetT(L)->array() OP ao.mScalar;						\
																							\
							lua_getref(L, td->mPushRef);/* mat, push_bool_matrix */			\
							lua_pushlightuserdata(L, &bm);/* mat, push_bool_matrix, bm */	\
							lua_call(L, 1, 1);	/* mat, bool_mat */							\
																							\
							return 1

//
#define EIGEN_AS_ARRAY(METHOD)	return NewMoveRet<R>(L, GetT(L)->array().METHOD())

//
#define EIGEN_AS_ARRAY_BOOL(METHOD)	return NewMoveRet<BoolMatrix>(L, GetT(L)->array().METHOD())

//
#define EIGEN_MATRIX_GET_MATRIX(METHOD)	return NewMoveRet<R>(L, GetT(L)->METHOD())

//
#define EIGEN_MATRIX_GET_MATRIX_COUNT(METHOD)	return NewMoveRet<R>(L, GetT(L)->METHOD(LuaXS::Int(L, 2)))

//
#define EIGEN_MATRIX_GET_MATRIX_COUNT_PAIR(METHOD)	return NewMoveRet<R>(L, GetT(L)->METHOD(LuaXS::Int(L, 2), LuaXS::Int(L, 3)))

//
#define EIGEN_MATRIX_GET_MATRIX_INDEX(METHOD)	return NewMoveRet<R>(L, GetT(L)->METHOD(LuaXS::Int(L, 2) - 1))

//
#define EIGEN_MATRIX_GET_MATRIX_INDEX_PAIR(METHOD)	return NewMoveRet<R>(L, GetT(L)->METHOD(LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3) - 1))

//
#define EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR(METHOD)	return NewMoveRet<R>(L, GetT(L)->METHOD(*GetR(L, 2)))

//
#define EIGEN_MATRIX_GET_MATRIX_SECOND_IS_MATRIX_OR_SCALAR(METHOD)	ArgObject<R> ao{L, 2};												\
																																		\
																	if (ao.mMat) return NewMoveRet<R>(L, GetT(L)->METHOD(*ao.mMat));	\
																																		\
																	else return NewMoveRet<R>(L, GetT(L)->METHOD(ao.mScalar))

//
#define EIGEN_MATRIX_GET_SCALAR(METHOD)	T::Scalar result = GetT(L)->METHOD();		\
																					\
										return LuaXS::PushArgAndReturn(L, result)

//
#define EIGEN_MATRIX_PAIR_VOID(METHOD)	GetT(L)->METHOD(*GetR(L, 2));	\
																		\
										return 0

#define EIGEN_MATRIX_REDUCE(METHOD)	ReductionOption how = GetReductionChoice(L, 2);									\
																													\
									if (how == eDefault)															\
									{																				\
										EIGEN_MATRIX_GET_SCALAR(METHOD);											\
									}																				\
																													\
									else																			\
									{																				\
										if (how == eColwise) return NewMoveRet<R>(L, GetT(L)->colwise().METHOD());	\
										else return NewMoveRet<R>(L, GetT(L)->rowwise().METHOD());					\
									}

//
#define EIGEN_MATRIX_SET_SCALAR(METHOD)	GetT(L)->METHOD(ArgObject<R>{}.AsScalar(L, 2));	\
																						\
										return 0

//
#define EIGEN_MATRIX_SET_SCALAR_CHAIN(METHOD)	GetT(L)->METHOD(ArgObject<R>{}.AsScalar(L, 2));	\
																								\
												return SelfForChaining(L)

//
#define EIGEN_MATRIX_ASSIGN_MATRIX_COUNT(METHOD)	GetT(L)->METHOD(LuaXS::Int(L, 2)) = *GetR(L, 3);\
																									\
													return 0

//
#define EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_PAIR(METHOD)	GetT(L)->METHOD(LuaXS::Int(L, 2), LuaXS::Int(L, 3)) = *GetR(L, 4);	\
																															\
														return 0

//
#define EIGEN_MATRIX_ASSIGN_MATRIX_INDEX(METHOD)	GetT(L)->METHOD(LuaXS::Int(L, 2) - 1) = *GetR(L, 3);\
																										\
													return 0

//
#define EIGEN_MATRIX_ASSIGN_MATRIX_INDEX_PAIR(METHOD)	GetT(L)->METHOD(LuaXS::Int(L, 2) - 1, LuaXS::Int(L, 3) - 1) = *GetR(L, 4);	\
																																	\
														return 0

//
#define EIGEN_MATRIX_PUSH_VALUE(METHOD)	return LuaXS::PushArgAndReturn(L, GetT(L)->METHOD())

//
#define EIGEN_MATRIX_VOID(METHOD)		GetT(L)->METHOD();	\
															\
										return 0

//
#define EIGEN_MATRIX_CHAIN(METHOD)		GetT(L)->METHOD();			\
																	\
										return SelfForChaining(L)



//
#define USING_COMPLEX_TYPE() using ComplexType = Eigen::Matrix<std::complex<R::Scalar>, Eigen::Dynamic, Eigen::Dynamic>
#define GET_COMPLEX_TYPE_DATA()	auto td = GetTypeData<ComplexType>(L);	\
																			\
									luaL_argcheck(L, td, 1, "Complex matrix type unavailable for cast")

#define PUSH_TYPED_DATA(item)	lua_getref(L, td->mPushRef);/* solver, ..., push_new_type */		\
									lua_pushlightuserdata(L, &item);/*solver, push_new_type, item */	\
									lua_call(L, 1, 1);	/* solver, conv_item */							\
																										\
									return 1

#define EIGEN_REAL_GET_COMPLEX(METHOD)	GET_COMPLEX_TYPE_DATA();			\
																			\
										ComplexType res = GetT(L)->METHOD();\
																			\
										PUSH_TYPED_DATA(res)

#define EIGEN_PUSH_AUTO_RESULT(METHOD)	auto res = GetT(L)->METHOD();					\
																						\
											return NewMoveRet<decltype(res)>(L, res)

//
#define EIGEN_REG(NAME, CALL)		#NAME, [](lua_State * L)	\
									{							\
										CALL;					\
									}

//
#define EIGEN_ARRAY_METHOD(NAME) EIGEN_REG(NAME, EIGEN_AS_ARRAY(NAME))
#define EIGEN_ARRAY_METHOD_BOOL(NAME) EIGEN_REG(NAME, EIGEN_AS_ARRAY_BOOL(NAME))
#define EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_METHOD(NAME) EIGEN_REG(NAME "Assign", EIGEN_MATRIX_ASSIGN_MATRIX_COUNT(NAME))
#define EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_PAIR_METHOD(NAME) EIGEN_REG(NAME "Assign", EIGEN_MATRIX_ASSIGN_MATRIX_COUNT_PAIR(NAME))
#define EIGEN_MATRIX_ASSIGN_MATRIX_INDEX_METHOD(NAME) EIGEN_REG(NAME "Assign", EIGEN_MATRIX_ASSIGN_MATRIX_INDEX(NAME))
#define EIGEN_MATRIX_ASSIGN_MATRIX_INDEX_PAIR_METHOD(NAME) EIGEN_REG(NAME "Assign", EIGEN_MATRIX_ASSIGN_MATRIX_INDEX_PAIR(NAME))
#define EIGEN_MATRIX_CHAIN_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_CHAIN(NAME))
#define EIGEN_MATRIX_GET_MATRIX_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_GET_MATRIX(NAME))
#define EIGEN_MATRIX_GET_MATRIX_COUNT_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_GET_MATRIX_COUNT(NAME))
#define EIGEN_MATRIX_GET_MATRIX_COUNT_PAIR_METHOD(NAME)	EIGEN_REG(NAME, EIGEN_MATRIX_GET_MATRIX_COUNT_PAIR(NAME))
#define EIGEN_MATRIX_GET_MATRIX_INDEX_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_GET_MATRIX_INDEX(NAME))
#define EIGEN_MATRIX_GET_MATRIX_INDEX_PAIR_METHOD(NAME)	EIGEN_REG(NAME, EIGEN_MATRIX_GET_MATRIX_INDEX_PAIR(NAME))
#define EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_GET_MATRIX_MATRIX_PAIR(NAME))
#define EIGEN_MATRIX_GET_MATRIX_SECOND_IS_MATRIX_OR_SCALAR_METHOD(NAME)	EIGEN_REG(NAME, EIGEN_MATRIX_GET_MATRIX_SECOND_IS_MATRIX_OR_SCALAR(NAME))
#define EIGEN_MATRIX_GET_SCALAR_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_GET_SCALAR(NAME))
#define EIGEN_MATRIX_PAIR_VOID_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_PAIR_VOID(NAME))
#define EIGEN_MATRIX_PUSH_VALUE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_PUSH_VALUE(NAME))
#define EIGEN_MATRIX_REDUCE_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_REDUCE(NAME))
#define EIGEN_MATRIX_SET_SCALAR_CHAIN_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_SET_SCALAR_CHAIN(NAME))
#define EIGEN_MATRIX_SET_SCALAR_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_SET_SCALAR(NAME))
#define EIGEN_MATRIX_VOID_METHOD(NAME) EIGEN_REG(NAME, EIGEN_MATRIX_VOID(NAME))
#define EIGEN_REL_OP_METHOD(NAME, OP) EIGEN_REG(NAME, EIGEN_REL_OP(OP))
#define EIGEN_REAL_GET_COMPLEX_METHOD(NAME)	EIGEN_REG(NAME, EIGEN_REAL_GET_COMPLEX(NAME))
#define EIGEN_PUSH_AUTO_RESULT_METHOD(NAME) EIGEN_REG(NAME, EIGEN_PUSH_AUTO_RESULT(NAME))