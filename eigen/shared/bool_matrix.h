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

#include "macros.h"
#include "types.h"
#include "utils.h"
#include "xprs.h"
#include "stock_ops.h"
#include "xpr_ops.h"

//
template<typename T, typename R = BoolMatrix> struct AttachBoolMatrixMethods : InstanceGetters<T, R> {
	AttachBoolMatrixMethods (lua_State * L)
	{
	#if defined(EIGEN_CORE) || defined(EIGEN_PLUGIN_BASIC)

		luaL_Reg methods[] = {
			{
				EIGEN_MATRIX_PUSH_VALUE_METHOD(all)
			}, {
				EIGEN_MATRIX_PUSH_VALUE_METHOD(any)
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
				EIGEN_MATRIX_PUSH_VALUE_METHOD(count)
			}, {
				"select", [](lua_State * L)
				{
					// For non-matrix objects such as maps, make a temporary and pass it
					// along to the select function. (Resolving this in the select logic
					// itself seems to cause major compilation slowdown.)
					ArgObjectR<R> bm{L, 1};

					if (!std::is_same<T, R>::value)
					{
						lua_pushlightuserdata(L, bm.mObject);	// mat, then, else, conv_mat
						lua_replace(L, 1);	// conv_mat, then, else
					}

					// Invoke the select logic appropriate to the supplied objects.
					int scalars = 0;

					if (!GetTypeData::GetName(L, 2)) ++scalars;	// mat, then, else[, name1]
					if (!GetTypeData::GetName(L, 3)) ++scalars;	// mat, then, else[, name1][, name2]

					luaL_argcheck(L, scalars < 2, 2, "Two scalars supplied to select()");
					luaL_argcheck(L, scalars > 0 || lua_equal(L, -2, -1), 2, "Mixed types supplied to select()");

					return GetTypeData::Select(L);	// selection
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
#if 0
/************************
* Boolean block methods *
************************/
template<typename U, int Rows, int Cols, bool InnerPanel> struct AttachMethods<Eigen::Block<U, Rows, Cols, InnerPanel>, BoolMatrix> : AttachBoolMatrixMethods<Eigen::Block<U, Rows, Cols, InnerPanel>> {
	AttachMethods (lua_State * L) : AttachBoolMatrixMethods<Eigen::Block<U, Rows, Cols, InnerPanel>>(L)
	{
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
template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols, int MapOptions, typename S> struct AttachMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>> : AttachBoolMatrixMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>> {
	AttachMethods (lua_State * L) : AttachBoolMatrixMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>>(L)
	{
	}
};

/****************************
* Boolean transpose methods *
****************************/
template<typename U> struct AttachMethods<Eigen::Transpose<U>, BoolMatrix> : AttachBoolMatrixMethods<Eigen::Transpose<U>> {
	AttachMethods (lua_State * L) : AttachBoolMatrixMethods<Eigen::Transpose<U>>(L)
	{
	}
};
#endif