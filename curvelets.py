import numpy as np
import functools
import math 
class directional_filter_bank():
    def _compute_decimation_factor_from_n_angles(self, n_angles):
        self.decimation_factor = 1
        while self.decimation_factor * 2 < n_angles:
            self.decimation_factor *= 2
        return self.decimation_factor
    
    def T_angle(self,x,y):
        result = np.zeros(x.shape)
        result = np.where(x >= abs(y), y/(x+ 1e-18), result)
        result = np.where(y >= abs(x), 2 - x/(y+ 1e-18), result)
        result = np.where(y <= - abs(x), -2 - x/(y+ 1e-18), result)
        result = np.where(x <=  - abs(y),   (y>=0) * (  4 + y/(x+ 1e-18)) \
                                         + (y< 0) * ( -4 + y/(x+ 1e-18))
                                       , result
                         )
        result = np.where(np.logical_and(x == 0, y == 0), 0, result)
        return result
    
    def _get_frame_functions(self, n_angles, nu_a = 0.3, nu_b = 0.2):
        poly = lambda t: np.polyval([- 5/32, 0, 21/32, 0, -35/32, 0, 35/32,1/2], t)
        beta_squared = lambda t: np.where(np.abs(t) < 1, poly(t), (t > 0).astype(float))
        safe_beta_squared = lambda t: np.clip(beta_squared(t), 0, 1) # when rounding error makes values out of [0,1] 
        beta = lambda t : np.sqrt(safe_beta_squared(t))
        w1_tilda = lambda t : beta((1 - abs(t))/nu_a)
        w0_tilda = lambda t : w1_tilda((2 * (1 + nu_a)) * t)
        w0 = lambda cx, cy : w0_tilda(cx)* w0_tilda(cy)
        w1 = lambda cx, cy : np.sqrt(1 - w0(cx, cy) ** 2) * w1_tilda(cx) * w1_tilda(cy)
        
        width_angle = 2/n_angles
        denominator = width_angle * nu_b
        v1_tilda = lambda t:   beta( ((width_angle - 1) - t)/ denominator ) \
                             * beta(    ( t + 1 )           / denominator )
        v_tilda  = lambda t, idx_angle : v1_tilda( t - width_angle * idx_angle )
        
        v = lambda cx,cy, idx_angle : v_tilda(self.T_angle(cx,cy), idx_angle)
        
        u_tilda = lambda cx, cy, idx_angle:  w1(cx, cy) * v(cx,cy, idx_angle)
        
        self.beta = beta
        self.w0 = w0
        self.w1 = w1
        self.v_tilda = v_tilda
        self.v = v
        self.u_tilda = u_tilda
        return self.w0, self.u_tilda
    
    def _compute_angular_filters(self,size_image, n_angles, border):
        graduation = np.arange(- size_image // 2, size_image // 2)
        x,y = np.meshgrid(graduation, graduation, indexing = 'ij')
        x = x / (size_image // 2)
        y = y / (size_image // 2)
        x = np.fft.fftshift(x)
        y = np.fft.fftshift(y)
        self.lowpass_filter  = np.expand_dims(self.w0(x,y), axis = 0)
        
        if border == "toric":
            self.angular_filters = np.array( [ [ [
                self.u_tilda(x + px,y + py, idx_angle) for px in [-2,0,2]
                                                 ]     for py in [-2,0,2]
                                               ]       for idx_angle in range(n_angles*2)
                                             ]
                                           )
            self.angular_filters = np.sum(self.angular_filters, axis = (1,2))
            
        elif border == "null":
            self.angular_filters = np.array( [ self.u_tilda(x ,y , idx_angle)
                                               for idx_angle in range(n_angles*2)
                                             ]
                                           )
        self.filters = np.concatenate( ( self.angular_filters, self.lowpass_filter ), axis = 0 )
        return self.filters
    
    def __init__(self, size_image, n_angles, nu_a = 0.3, nu_b = 0.2, border="null"):  
        self.n_angles = n_angles
        self.nu_a = nu_a
        self.nu_b = nu_b
        self.border = border
        self._compute_decimation_factor_from_n_angles(n_angles) 
        self._get_frame_functions(n_angles, nu_a = 0.3, nu_b = 0.2)
        self._compute_angular_filters(size_image, n_angles, border)

        
    def _decimation(self,arr, coef, axis):
        return functools.reduce( lambda a,b : a+b, 
                                 np.split( arr  , 
                                           coef , 
                                           axis = axis
                                         )
                               )
        
    def __call__(self,image):
        fft = np.fft.fft2(image, norm = "ortho")
        ndims_image = len(fft.shape)
        ndims_filter = 3
        axis_filter = ndims_image - 2
        axis_real_imag = axis_filter + 1
        
        expanded_filters = self.filters
        for _ in range(axis_filter):
            expanded_filters = np.expand_dims(expanded_filters, axis = 0)
        fft = np.expand_dims(fft, axis = axis_filter)
        
        filtered_fft = fft * expanded_filters
        
        filtered_fft = np.expand_dims( filtered_fft, axis_real_imag )
        
        
        vdirectional_filtered, hdirectional_filtered, lowfreq_filtered  = \
                np.split(     filtered_fft, 
                              [self.n_angles, 2* self.n_angles], 
                              axis = axis_filter 
                        )
        lowfreq_filtered = self._decimation(lowfreq_filtered, 2 , -1)
        lowfreq_filtered = self._decimation(lowfreq_filtered, 2 , -2)
        vdirectional_filtered = self._decimation(vdirectional_filtered, 2, -2)
        vdirectional_filtered = self._decimation(vdirectional_filtered, self.decimation_factor , -1)
        hdirectional_filtered = self._decimation(hdirectional_filtered, self.decimation_factor , -2)
        hdirectional_filtered = self._decimation(hdirectional_filtered, 2 , -1)
        
        hdirectional_filtered = np.fft.ifft2(hdirectional_filtered, norm = "ortho")
        vdirectional_filtered = np.fft.ifft2(vdirectional_filtered, norm = "ortho")
        lowfreq_filtered = np.fft.ifft2(lowfreq_filtered, norm = "ortho")
        
        hdirectional_filtered = np.concatenate( ( hdirectional_filtered.real, 
                                                  hdirectional_filtered.imag
                                                ), 
                                                axis = axis_real_imag
                                              )
        vdirectional_filtered = np.concatenate( ( vdirectional_filtered.real, 
                                                  vdirectional_filtered.imag
                                                ), 
                                                axis = axis_real_imag
                                              )
        
        hdirectional_filtered = hdirectional_filtered * math.sqrt(2)
        vdirectional_filtered = vdirectional_filtered * math.sqrt(2)
        lowfreq_filtered = lowfreq_filtered.real
        
        return (lowfreq_filtered, vdirectional_filtered, hdirectional_filtered)
    
    def reconstruction(self, lowfreq_filtered, vdirectional_filtered, hdirectional_filtered):
        ndims_image = len(lowfreq_filtered.shape) - 2
        axis_filter = ndims_image - 2
        axis_real_imag = axis_filter + 1
        
        expanded_filters = self.filters
        for _ in range(axis_filter):
            expanded_filters = np.expand_dims(expanded_filters, axis = 0)
        
        get_real_part = lambda arr: np.take(arr, 0, axis = axis_real_imag)
        get_imag_part = lambda arr: np.take(arr, 1, axis = axis_real_imag)
        to_complex    = lambda arr: get_real_part(arr) + 1j * get_imag_part(arr)
        
        
        lowfreq_filtered = np.fft.fft2(lowfreq_filtered, norm = "ortho")
        lowfreq_filtered = np.squeeze(lowfreq_filtered, axis = axis_real_imag)
        
        
        hdirectional_filtered = np.fft.fft2(  to_complex(hdirectional_filtered), norm = "ortho" ) /math.sqrt(2)
        
        vdirectional_filtered = np.fft.fft2(  to_complex(vdirectional_filtered), norm = "ortho") /math.sqrt(2)
        
        lowfreq_filtered = np.tile(lowfreq_filtered, [1] * (ndims_image - 1) + [2,2]) 
        hdirectional_filtered = np.tile( hdirectional_filtered, [1] * (ndims_image - 1) + [self.decimation_factor,2] )
        vdirectional_filtered = np.tile( vdirectional_filtered, [1] * (ndims_image - 1) + [2,self.decimation_factor] )
        
        filtered_fft = np.concatenate((vdirectional_filtered, hdirectional_filtered, lowfreq_filtered), axis = axis_filter)
        filtered_fft = filtered_fft * expanded_filters
        
        hf_filtered, lowfreq_filtered = np.split(filtered_fft, [2*self.n_angles], axis = axis_filter)
        lowfreq_filtered = np.squeeze(lowfreq_filtered, axis = axis_filter)
        hf_filtered =  np.sum( hf_filtered, axis = axis_filter)
        
        
        hf_filtered_flipped = np.flip(hf_filtered, axis =(-1))
        hf_filtered_flipped = np.roll(hf_filtered_flipped, 1, axis =(-1))
        hf_filtered_flipped = np.flip(hf_filtered_flipped, axis =(-2))
        hf_filtered_flipped = np.roll(hf_filtered_flipped, 1, axis =(-2))
        

        hf_filtered = hf_filtered + np.conj(hf_filtered_flipped)
        return np.fft.ifft2(hf_filtered + lowfreq_filtered, norm = "ortho").real
 
class curvelet_transform():
    def __init__(self, size_image, nums_angles, nu_a = 0.3, nu_b = 0.2):
        self._directional_filter_banks = []
        border = "toric"
        size = size_image
        self.nums_angles = list(nums_angles)
        for num_angle in reversed(self.nums_angles):
            self._directional_filter_banks += [directional_filter_bank(size, num_angle, nu_a, nu_b, border)]
            size = size/2
            border="null"
        
    
    def __call__(self, image):
        result = [np.expand_dims(image, axis = (-4,-3))]
        for dir_filt_bank in self._directional_filter_banks:
            result = list(dir_filt_bank(np.squeeze(result[0], axis = (-4,-3)))) + result[1:]
        return result
    
    def inverse(self, transform):
        result = transform
        for dir_filt_bank in reversed(self._directional_filter_banks):
            result = [dir_filt_bank.reconstruction(result[0], result[1], result[2])] + result[3:]
            result[0] = np.expand_dims(result[0], axis = (-4,-3))
        return np.squeeze(result[0], axis = (-4,-3))
        
        
        
    
    
            
    
        