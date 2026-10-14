
import numpy as np

def NelderMead_mapping_parameter ( par_original, Params_LowerBound, Params_UpperBound):
    # 给NelderMead 单纯形方法加上下限的一个方法，如果参数值超过边界，以界线为对称轴对参数值做轴对称变换等
    # To map the parameters inside the parameter boundary
    # input: 'par_original' original parameters,
    # output: 'new_par', mapped parameters, within the upper and lower bounds
    # this function follows QJ Wang 's method, and Jian Jie's matlab code
  
  
    nparams=len( par_original)
    new_par=np.zeros_like(par_original)#变换后的参数
    
    for  ipar in range(nparams):#遍历各个参数
    
        LowerBound = Params_LowerBound[ipar]
        UpperBound = Params_UpperBound[ipar]

        flag_inf_lowerbound = np.isneginf(LowerBound)
        flag_inf_upperbound = np.isposinf(UpperBound )

        if ( flag_inf_lowerbound & flag_inf_upperbound ):#上下界均是无穷，不用变换
            new_par[ipar] = par_original[ipar] 
        
        elif ( flag_inf_lowerbound & (~flag_inf_upperbound) ):#上界不是无穷
            new_par[ipar] = UpperBound - np.abs(par_original[ipar] - UpperBound )

        elif ( (~flag_inf_lowerbound) & flag_inf_upperbound ):#下界不是无穷
            new_par[ipar] = LowerBound + np.abs(par_original[ipar] - LowerBound )


        else: #上下界均不是无穷

            # Step 1: checking lower boundary,flip the values
            new_par[ipar] = (2*LowerBound-par_original[ipar]) if (par_original[ipar] < LowerBound)  else   par_original[ipar]
            # Step 2: move
            new_par[ipar] = new_par[ipar] - 2 *(UpperBound-LowerBound) * np.floor( (new_par[ipar]-LowerBound) /(2 *(UpperBound-LowerBound)) ) 
            # Step 3: checking upper boundary,flip the values
            new_par[ipar] = (2 *UpperBound-new_par[ipar]) if (new_par[ipar] > UpperBound)  else  new_par[ipar]
  
     
  
    return new_par 