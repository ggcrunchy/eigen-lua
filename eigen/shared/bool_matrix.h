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
#include "xprs.h"
#include "xpr_ops.h"
#include "utils/LuaEx.h"
#include <string>
#include <Eigen/Eigen>

/*********************
* BoolMatrix methods *
*********************/
template<> struct AttachMethods<BoolMatrix> {
	static BoolMatrix * GetT (lua_State * L, int arg = 1)
	{
		return LuaXS::CheckUD<BoolMatrix>(L, arg, "eigen.bool");
	}

	AttachMethods (lua_State * L)
	{
	#if defined(EIGEN_CORE) || defined(EIGEN_PLUGIN_BASIC)

		luaL_Reg methods[] = {
			{
				"all", [](lua_State * L)
				{
					switch (GetVectorwiseOption(L, 2))
					{
					case eColwise:
						return NewRet<BoolMatrix>(L, GetT(L)->colwise().all());
					case eRowwise:
						return NewRet<BoolMatrix>(L, GetT(L)->rowwise().all());
					default: // GetReductionChoice() will trap anything else
						EIGEN_MATRIX_PUSH_VALUE(all);
					}
				}
			}, {
				"any", [](lua_State * L)
				{
					switch (GetVectorwiseOption(L, 2))
					{
					case eColwise:
						return NewRet<BoolMatrix>(L, GetT(L)->colwise().any());
					case eRowwise:
						return NewRet<BoolMatrix>(L, GetT(L)->rowwise().any());
					default: // GetReductionChoice() will trap anything else
						EIGEN_MATRIX_PUSH_VALUE(any);
					}
				}
			}, {
				"band", [](lua_State * L)
				{
					return NewRet<BoolMatrix>(L, *GetT(L) && *GetT(L, 2));
				}
			}, {
				"bor", [](lua_State * L)
				{
					return NewRet<BoolMatrix>(L, *GetT(L) || *GetT(L, 2));
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
					auto how = GetVectorwiseOption(L, 2);

					if (how == eNotVectorwise) EIGEN_MATRIX_PUSH_VALUE(count);

					else
					{
						auto td = GetTypeData<Eigen::MatrixXi>(L, eFetchIfMissing);

						luaL_argcheck(L, td, 2, "Column- or row-wise count() requires int matrices");

						Eigen::MatrixXi im;

						if (how == eColwise) im = GetT(L)->colwise().count();
						else im = GetT(L)->rowwise().count();

						PUSH_TYPED_DATA(im);
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
					TypeData * types[] = {
						GetTypeData<Eigen::MatrixXi>(L, eFetchIfMissing),
						GetTypeData<Eigen::MatrixXf>(L, eFetchIfMissing),
						GetTypeData<Eigen::MatrixXd>(L, eFetchIfMissing),
						GetTypeData<Eigen::MatrixXcf>(L, eFetchIfMissing),
						GetTypeData<Eigen::MatrixXcd>(L, eFetchIfMissing)
					};

					for (auto cur : types)
					{
						if (cur && (LuaXS::IsType(L, cur->GetName(), 1) || LuaXS::IsType(L, cur->GetName(), 2)))
						{
							lua_settop(L, 3);	// mat, then, else
							lua_getref(L, cur->mSelectRef);	// mat, then, else, select
							lua_insert(L, 1);	// select, mat, then, else
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

		XprOps<BoolMatrix> xo{L};

	#endif
	}
};