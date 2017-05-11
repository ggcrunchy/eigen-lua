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
#include "utils/LuaEx.h"
#include <Eigen/Eigen>

//
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> BoolMatrix;

//
template<> struct AttachMethods<BoolMatrix> {
	static BoolMatrix * GetT (lua_State * L, int arg = 1)
	{
		return LuaXS::CheckUD<BoolMatrix>(L, arg, "eigen.bool");
	}

	AttachMethods (lua_State * L)
	{
		luaL_Reg methods[] = {
			{
				"all", [](lua_State * L)
				{
					switch (GetReductionChoice(L, 2))
					{
					case eDefault:
						EIGEN_MATRIX_PUSH_VALUE(all);
					case eColwise:
						return NewMoveRet<BoolMatrix>(L, GetT(L)->colwise().all());
					default: // GetReductionChoice() will trap anything else
						return NewMoveRet<BoolMatrix>(L, GetT(L)->rowwise().all());
					}
				}
			}, {
				"any", [](lua_State * L)
				{
					switch (GetReductionChoice(L, 2))
					{
					case eDefault:
						EIGEN_MATRIX_PUSH_VALUE(any);
					case eColwise:
						return NewMoveRet<BoolMatrix>(L, GetT(L)->colwise().any());
					default: // GetReductionChoice() will trap anything else
						return NewMoveRet<BoolMatrix>(L, GetT(L)->rowwise().any());
					}
				}
			}, {
				"band", [](lua_State * L)
				{
					return NewMoveRet<BoolMatrix>(L, *GetT(L) && *GetT(L, 2));
				}
			}, {
				"bor", [](lua_State * L)
				{
					return NewMoveRet<BoolMatrix>(L, *GetT(L) || *GetT(L, 2));
				}
			}, {
				"__call", [](lua_State * L)
				{
					BoolMatrix & m = *GetT(L);
					int a = LuaXS::Int(L, 2) - 1;
					bool result;

					if (lua_gettop(L) == 2)
					{
						CheckVector(L, m, 1);

						result = m.cols() == 1 ? m(a, 0) : m(0, a);
					}

					else result = m(a, LuaXS::Int(L, 3) - 1);

					return LuaXS::PushArgAndReturn(L, result);
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(cols)
			}, {
				"count", [](lua_State * L)
				{
					auto how = GetReductionChoice(L, 2);

					if (how == eDefault) EIGEN_MATRIX_PUSH_VALUE(count);

					else
					{
						auto td = GetTypeData<Eigen::MatrixXi>(L);

						luaL_argcheck(L, td, 2, "Column- or row-wise count() requires int matrices");

						Eigen::MatrixXi im;

						if (how == eColwise) im = GetT(L)->colwise().count();
						else im = GetT(L)->rowwise().count();

						lua_getref(L, td->mPushRef);	// m, how, push
						lua_pushlightuserdata(L, &im);	// m, how, push, counts
						lua_call(L, 1, 1);	// m, how, count_im

						return 1;
					}
				}
			}, {
				"__eq", [](lua_State * L)
				{
					return LuaXS::PushArgAndReturn(L, *GetT(L) == *GetT(L, 2));
				}
			}, {
				"__gc", LuaXS::TypedGC<BoolMatrix>
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(rows)
			}, {
				"select", [](lua_State * L)
				{
					const char * names[] = {
						FullName<Eigen::MatrixXi>(),
						FullName<Eigen::MatrixXf>(),
						FullName<Eigen::MatrixXd>(),
						FullName<Eigen::MatrixXcf>(),
						FullName<Eigen::MatrixXcd>(),
						nullptr
					};

					const char * data_keys[] = {
						FullName<Eigen::MatrixXi>(true),
						FullName<Eigen::MatrixXf>(true),
						FullName<Eigen::MatrixXd>(true),
						FullName<Eigen::MatrixXcf>(true),
						FullName<Eigen::MatrixXcd>(true)
					};

					for (int i = 0; names[i]; ++i)
					{
						if (LuaXS::IsType(L, names[i], 1) || LuaXS::IsType(L, names[i], 2))
						{
							lua_settop(L, 3);	// mat, then, else
							lua_getfield(L, LUA_REGISTRYINDEX, data_keys[i]);	// mat, then, else, typed_data
							lua_getref(L, LuaXS::UD<TypeData>(L, -1)->mSelectRef);	// mat, then, else, typed_data, select
							lua_insert(L, 1);	// select, mat, then, else, typed_data
							lua_pop(L, 1);	// select, mat, then, else
							lua_call(L, 3, 1);	// m

							return 1;
						}
					}

					luaL_error(L, "No typed matrix provided to select");

					return 0;
				}
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(size)
			}, {
				"__tostring", [](lua_State * L)
				{
					return Print(L, *GetT(L));
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);
	}
};