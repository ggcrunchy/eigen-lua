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

#include <Eigen/Eigen>
#include "utils/LuaEx.h"
#include "config.h"
#include "types.h"
#include "utils.h"
#include "bool_matrix.h"
#include "common.h"
#include "complex_dependent.h"
#include "non_int.h"
#include <type_traits>

// Common matrix methods attachment body.
template<typename T, typename R> struct AttachMatrixMethods {
	static_assert(std::is_same<typename T::Scalar, typename R::Scalar>::value, "Types have mixed scalars");

	AttachMatrixMethods (lua_State * L)
	{
		CommonMethods<T, R>{L};
		ComplexDependentMethods<T, R>{L};
		NonIntMethods<T, R>{L};
	}
};

/*****************
* Matrix methods *
*****************/
template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols, typename R> struct AttachMethods<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, R> : AttachMatrixMethods<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, R> {
	AttachMethods (lua_State * L) : AttachMatrixMethods<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, R>(L)
	{
	}
};

/************************
* Mapped matrix methods *
************************/
template<typename T, int Rows, int Cols, int Options, int MaxRows, int MaxCols, int MapOptions, typename S, typename R> struct AttachMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>, R> : AttachMatrixMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>, R> {
	AttachMethods (lua_State * L) : AttachMatrixMethods<Eigen::Map<Eigen::Matrix<T, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, S>, R>(L)
	{
	}
};

// TODO: transpose, etc.