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
#include "stock_ops.h"
#include "xpr_ops.h"
#include "utils/LuaEx.h"
#include <string>
#include <type_traits>
#include <Eigen/Eigen>

//
template<typename T, typename R = BoolMatrix> struct AttachBoolMatrixMethods {
	ADD_INSTANCE_GETTERS()

	AttachBoolMatrixMethods (lua_State * L)
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
				"count", [](lua_State * L)
				{
					auto how = GetVectorwiseOption(L, 2);

					if (how == eNotVectorwise) EIGEN_MATRIX_PUSH_VALUE(count);

					else
					{
						auto td = GetTypeData<Eigen::MatrixXi>(L, TypeData::eFetchIfMissing);

						luaL_argcheck(L, td, 2, "Column- or row-wise count() requires int matrices");

						Eigen::MatrixXi im;

						if (how == eColwise) im = GetT(L)->colwise().count();
						else im = GetT(L)->rowwise().count();

						PUSH_TYPED_DATA(im);
					}
				}
			}, {
				"select", [](lua_State * L)
				{
					TypeData * types[] = {
						GetTypeData<Eigen::MatrixXi>(L, TypeData::eFetchIfMissing),
						GetTypeData<Eigen::MatrixXf>(L, TypeData::eFetchIfMissing),
						GetTypeData<Eigen::MatrixXd>(L, TypeData::eFetchIfMissing),
						GetTypeData<Eigen::MatrixXcf>(L, TypeData::eFetchIfMissing),
						GetTypeData<Eigen::MatrixXcd>(L, TypeData::eFetchIfMissing)
					};

					for (auto cur : types)
					{
						if (cur && (LuaXS::IsType(L, cur->GetName(), 1) || LuaXS::IsType(L, cur->GetName(), 2)))
						{
							lua_settop(L, 3);	// mat, then, else
							lua_getref(L, cur->mSelectRef);	// mat, then, else, select

							// For non-matrix objects such as maps, make a temporary and pass it along to the
							// select function. (Resolving this in the select logic itself seems to cause major
							// compilation slowdown.)
							ArgObjectR<R> bm{L, 1};

							if (!std::is_same<T, R>::value)
							{
								lua_pushlightuserdata(L, bm.mObject);	// mat, then, else, select, conv_mat
								lua_replace(L, 1);	// conv_mat, then, else, select
							}

							lua_insert(L, 1);	// select, mat, then, else
							lua_call(L, 3, 1);	// m

							return 1;
						}
					}

					luaL_error(L, "No typed matrix provided to select");

					return 0;
				}
			},
			{ nullptr, nullptr }
		};

		luaL_register(L, nullptr, methods);

		StockOps<BoolMatrix> so{L};
		WriteOps<BoolMatrix> wo{L};
		XprOps<BoolMatrix> xo{L};

	#endif
	}
};

/*********************
* BoolMatrix methods *
*********************/
template<int Rows, int Cols, int Options, int MaxRows, int MaxCols> struct AttachMethods<Eigen::Matrix<bool, Rows, Cols, Options, MaxRows, MaxCols>> : AttachBoolMatrixMethods<Eigen::Matrix<bool, Rows, Cols, Options, MaxRows, MaxCols>> {
	AttachMethods (lua_State * L) : AttachBoolMatrixMethods<Eigen::Matrix<bool, Rows, Cols, Options, MaxRows, MaxCols>>(L)
	{
	}
};

/********************************
* Mapped boolean matrix methods *
********************************/
template<int Rows, int Cols, int Options, int MaxRows, int MaxCols, int MapOptions, typename S> struct AttachMethods<Eigen::Map<Eigen::Matrix<bool, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>> : AttachBoolMatrixMethods<Eigen::Map<Eigen::Matrix<bool, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>> {
	AttachMethods (lua_State * L) : AttachBoolMatrixMethods<Eigen::Map<Eigen::Matrix<bool, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>>(L)
	{
	}
};