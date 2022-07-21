from cmath import pi
from Hardware.Accumulator_TIA import Accumulator_TIA
from Hardware.BtoS import BtoS
from Hardware.MRR import MRR
from Hardware.Serializer import Serializer
from Hardware.eDram import EDram
from Hardware.ADC import ADC
from Hardware.DAC import DAC
from Hardware.PD import PD
from Hardware.TIA import TIA
from Hardware.VCSEL import VCSEL
from Hardware.io_interface import IOInterface
from Hardware.bus import Bus
from Hardware.router import Router
from Hardware.Activation import Activation


class Metrics:

    def __init__(self):
        self.eDram = EDram()
        self.adc = ADC()
        self.dac = DAC()
        self.pd = PD()
        self.tia = TIA()
        self.mrr = MRR()
        self.io_interface = IOInterface()
        self.bus = Bus()
        self.router = Router()
        self.activation = Activation()
        self.serializer = Serializer()
        self.accum_tia = Accumulator_TIA()
        self.b_to_s = BtoS()
        self.vcsel = VCSEL()
        self.laser_power_per_wavelength = 1.274274986e-3
        self.wall_plug_efficiency = 5  # 20%
        self.thermal_tuning_latency = 4000e-9
        self.photonic_adder = 1060+5.12+200 
        

    def get_hardware_utilization(self, utilized_rings, idle_rings):

        return (utilized_rings/(utilized_rings+idle_rings))*100

    def get_dynamic_energy(self, accelerator, utilized_rings):

        total_energy = 0
        # * For each vdp in accelerator the number of calls gives the number of times eDRam is called
        # * The dynamic energy of ADC, DAC , MRR = no of rings utilized*their eneergy
        # * PD and TIA energy = vdp calls * no of vdp elements in each VDP

        for vdp in accelerator.vdp_units_list:
            eDram_energy = vdp.calls_count*self.eDram.energy
            pd_energy = vdp.calls_count*vdp.get_element_count()*self.pd.energy
            tia_energy = vdp.calls_count*vdp.get_element_count()*self.tia.energy
            total_energy += eDram_energy+pd_energy+tia_energy
        if accelerator.acc_type == 'ANALOG':
            dac_energy = self.dac.energy*utilized_rings
            total_energy = total_energy+dac_energy
        elif accelerator.acc_type == 'STOCHASTIC':
            serializer_energy = self.serializer.energy*utilized_rings
            total_energy = total_energy + serializer_energy
        elif accelerator.acc_type == 'ROBIN':
            vcsel_energy = vdp.calls_count*self.vcsel.energy
            total_energy = total_energy + vcsel_energy
            
        adc_energy = self.adc.energy*utilized_rings
        mrr_energy = self.mrr.energy*utilized_rings
        total_energy += adc_energy+mrr_energy
        return total_energy

    def get_total_latency(self, latencylist):
        total_latency = sum(latencylist)+self.thermal_tuning_latency
        return total_latency

    def get_static_power(self, accelerator):

        total_power = 0
        vdp_power = 0
        for vdp in accelerator.vdp_units_list:
            # * adding no of comb switches  to the vdp element
            element_size = vdp.vdp_element_list[0].element_size
            elements_count = vdp.get_element_count()

            if vdp.vdp_type == 'AMM':
                if accelerator.acc_type == 'ANALOG':
                    no_of_adc = elements_count*element_size
                    no_of_dac = 2*elements_count*element_size
                    no_of_pd = elements_count
                    no_of_tia = elements_count
                    no_of_mrr = 2*elements_count*element_size+element_size
                    laser_power = self.laser_power_per_wavelength*elements_count*element_size
                    power_params = {}
                    power_params['adc'] = no_of_adc*self.adc.power
                    power_params['dac'] = no_of_dac*self.dac.power
                    power_params['pd'] = no_of_pd*self.pd.power
                    power_params['tia'] = no_of_tia*self.tia.power
                    power_params['mrr'] = no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to)
                    power_params['laser_power'] = laser_power
                    vdp_power += no_of_adc*self.adc.power + no_of_dac*self.dac.power + no_of_pd*self.pd.power + no_of_tia * \
                        self.tia.power + no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to) + \
                        laser_power*self.wall_plug_efficiency
                elif accelerator.acc_type == 'STOCHASTIC':
                    no_of_adc = elements_count*element_size
                    no_of_btos = 2*elements_count*element_size
                    no_of_pd = elements_count
                    no_of_accumulator = elements_count
                    no_of_serializers = 2*elements_count*element_size
                    no_of_mrr = 2*elements_count*element_size+element_size
                    laser_power = self.laser_power_per_wavelength*elements_count*element_size
                    power_params = {}
                    power_params['adc'] = no_of_adc*self.adc.power
                    power_params['btos'] = no_of_btos*self.b_to_s.power
                    power_params['pd'] = no_of_pd*self.pd.power
                    power_params['accumulator_tia'] = no_of_accumulator * \
                        self.accum_tia.power
                    power_params['mrr'] = no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to)
                    power_params['serializer'] = no_of_serializers * \
                        self.serializer.power
                    power_params['laser_power'] = laser_power
                    vdp_power += no_of_adc*self.adc.power+no_of_btos*self.b_to_s.power+no_of_pd*self.pd.power+no_of_accumulator*self.tia.power + \
                        no_of_mrr*(self.mrr.power_eo+self.mrr.power_to) + no_of_serializers * \
                        self.serializer.power + laser_power*self.wall_plug_efficiency
                elif accelerator.acc_type == 'ROBIN':
                    no_of_adc = elements_count*element_size
                    no_of_dac = 2*elements_count*element_size
                    no_of_pd = elements_count
                    no_of_tia = elements_count
                    no_of_VCSEL = elements_count
                    no_of_mrr = 2*elements_count*element_size+element_size
                    laser_power = self.laser_power_per_wavelength*elements_count*element_size
                    power_params = {}
                    power_params['adc'] = no_of_adc*self.adc.power
                    power_params['dac'] = no_of_dac*self.dac.power
                    power_params['pd'] = no_of_pd*self.pd.power
                    power_params['tia'] = no_of_tia*self.tia.power
                    power_params['mrr'] = no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to)
                    power_params['laser_power'] = laser_power
                    vdp_power += no_of_adc*self.adc.power + no_of_dac*self.dac.power + no_of_pd*self.pd.power + no_of_tia * \
                        self.tia.power + no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to) + \
                        laser_power*self.wall_plug_efficiency + no_of_VCSEL*self.vcsel.power
                elif accelerator.acc_type == 'ONNA':
                    no_of_adc = elements_count*element_size
                    no_of_dac = elements_count*element_size
                    no_of_pd = elements_count
                    no_of_tia = elements_count
                    no_of_VCSEL = elements_count
                    no_of_mrr = elements_count*element_size+element_size
                    laser_power = self.laser_power_per_wavelength*elements_count*element_size
                    power_params = {}
                    power_params['adc'] = no_of_adc*self.adc.power
                    power_params['dac'] = no_of_dac*self.dac.power
                    power_params['pd'] = no_of_pd*self.pd.power
                    power_params['tia'] = no_of_tia*self.tia.power
                    power_params['mrr'] = no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to)
                    power_params['laser_power'] = laser_power
                    vdp_power += no_of_adc*self.adc.power + no_of_dac*self.dac.power + no_of_pd*self.pd.power + no_of_tia * \
                        self.tia.power + no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to) + \
                        laser_power*self.wall_plug_efficiency + no_of_VCSEL*self.vcsel.power
                elif accelerator.acc_type == 'LIGHTBULB':
                    no_of_adc = elements_count*element_size
                    no_of_dac = 3*elements_count*element_size
                    no_of_pd = elements_count
                    no_of_tia = elements_count
                    no_of_mrr = 3*elements_count*element_size+element_size
                    laser_power = self.laser_power_per_wavelength*elements_count*element_size
                    power_params = {}
                    power_params['adc'] = no_of_adc*self.adc.power
                    power_params['dac'] = no_of_dac*self.dac.power
                    power_params['pd'] = no_of_pd*self.pd.power
                    power_params['tia'] = no_of_tia*self.tia.power
                    power_params['mrr'] = no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to)
                    power_params['laser_power'] = laser_power
                    vdp_power += no_of_adc*self.adc.power + no_of_dac*self.dac.power + no_of_pd*self.pd.power + no_of_tia * \
                        self.tia.power + no_of_mrr * \
                        (self.mrr.power_eo+self.mrr.power_to) + \
                        laser_power*self.wall_plug_efficiency 

            elif vdp.vdp_type == 'MAM':
                no_of_adc = elements_count*element_size
                no_of_dac = elements_count*element_size
                no_of_pd = elements_count
                no_of_tia = elements_count
                no_of_mrr = 2*elements_count*element_size+element_size
                laser_power = self.laser_power_per_wavelength*elements_count*element_size
                power_params = {}
                power_params['adc'] = no_of_adc*self.adc.power
                power_params['dac'] = no_of_dac*self.dac.power
                power_params['pd'] = no_of_pd*self.pd.power
                power_params['tia'] = no_of_tia*self.tia.power
                power_params['mrr'] = no_of_mrr * \
                    (self.mrr.power_eo+self.mrr.power_to)
                power_params['laser_power'] = laser_power

                vdp_power += no_of_adc*self.adc.power + no_of_dac*self.dac.power + no_of_pd*self.pd.power + no_of_tia * \
                    self.tia.power + no_of_mrr * \
                    (self.mrr.power_eo+self.mrr.power_to) + \
                    laser_power*self.wall_plug_efficiency

                # print("VDP Power ", vdp_power)
            pheripheral_power_params = {}
            pheripheral_power_params['io'] = self.io_interface.power
            pheripheral_power_params['bus'] = self.bus.power
            pheripheral_power_params['eram'] = self.eDram.power
            pheripheral_power_params['router'] = self.router.power
            pheripheral_power_params['activation'] = self.activation.power
            total_power += self.io_interface.power + self.activation.power + \
                self.router.power + self.bus.power + vdp_power + self.eDram.power
            print("Total Power ", total_power)
        return total_power

    def get_total_area(self, TYPE, X, Y, N, M, N_FC, M_FC, reconfig_list=[], acc_type='ANALOG'):
        pitch = 5  # um
        radius = 4.55  # um
        S_A_area = 0.00003  # mm2
        eDram_area = 0.166  # mm2
        max_pool_area = 0.00024  # mm2
        sigmoid = 0.0006  # mm2
        router = 0.151  # mm2
        bus = 0.009  # mm2
        splitter = 0.005  # mm2
        pd = 1.40625 * 1e-5  # mm2
        adc = self.adc.area  # mm2
        dac = self.dac.area  # mm2
        io_interface = 0.0244  # mm2
        serializer = 5.9*1e-3  # mm2

        if TYPE == 'MAM':
            print('MAM')
            # premux = 30  # um
            # WDM = 2 * pitch + 2 * radius
            # preweight = 130
            # weightblk = N * radius + N * pitch + 2 * pitch
            # width = premux + WDM + preweight + weightblk
            # height_N = 4 * pitch + N * (radius + pitch)
            # height_M = 4 * pitch + M * (radius + pitch)
            # if height_M > height_N:
            #     height = height_M
            # else:
            #     height = height_N
            # cnn_vdp_unit_area = height * width * 1e-6  # mm2
            # print('CNN unit area ', cnn_vdp_unit_area)
            # fc_weightblk = N_FC * radius + N_FC * pitch + 2 * pitch
            # fc_width = premux + WDM + preweight + fc_weightblk
            # fc_height = 3 * pitch + N_FC * (radius + pitch)
            # fc_vdp_unit_area = fc_width * fc_height * 1e-6  # mm2
            mrr_area = (3.14*(radius**2)*1e-6)

            cnn_vdp_unit_area = mrr_area*N*M + M*mrr_area

            splitter_area = M * splitter

            splitter_area_FC = 0
            pd_area = (M) * pd

            adc_area = M * (N+1) * adc

            dac_area = M * (N) * dac

            total_cnn_units_area = X * \
                (cnn_vdp_unit_area + pd_area + splitter_area + adc_area + dac_area)

            total_area = total_cnn_units_area + S_A_area + \
                eDram_area + sigmoid + router + bus + max_pool_area+io_interface
            # print('N', N)
            # print('M', M)
            # print('total_cnn_units_area', total_cnn_units_area)
            # print('S_A_area', S_A_area)
            # print('eDram_area', eDram_area)
            # print('sigmoid', sigmoid)
            # print('router', router)
            # print('bus', bus)
            # print('max_pool_area', max_pool_area)
            # print('io_interface', io_interface)
            return total_area
        elif TYPE == 'AMM':
            print('AMM')
            # premux = 30  # um
            # WDM = 2 * pitch + 2 * radius
            # preweight = 130
            # weightblk = 2*(N * radius + N * pitch) + 2 * pitch
            # width = premux + WDM + preweight + weightblk
            # height_N = 4 * pitch + N * (radius + pitch)
            # height_M = 4 * pitch + M * (radius + pitch)
            # if height_M > height_N:
            #     height = height_M
            # else:
            #     height = height_N
            # cnn_vdp_unit_area = height * width * 1e-6  # mm2
            # print('CNN unit area ', cnn_vdp_unit_area)
            # fc_weightblk = 2*(N_FC * radius + N_FC * pitch) + 2 * pitch
            # fc_width = premux + WDM + preweight + fc_weightblk
            # fc_height = 3 * pitch + N_FC * (radius + pitch)
            # fc_vdp_unit_area = fc_width * fc_height * 1e-6  # mm2
            mrr_area = (3.14*(radius**2)*1e-6)
            splitter_area = M * splitter

            pd_area = (M) * pd

            adc_area = M * (N) * adc
            
            if acc_type == 'STOCHASTIC':
                cnn_vdp_unit_area = mrr_area*(2*N)*M
                serializer_area = M * (N) * serializer
                serializer_area_fc = M_FC * N_FC * serializer
                dac_area = M * (2*N) * dac
                total_cnn_units_area = X * \
                    (cnn_vdp_unit_area + pd_area +
                     splitter_area + adc_area + serializer_area)
                total_area = total_cnn_units_area + S_A_area + \
                eDram_area + sigmoid + router + bus + max_pool_area+io_interface
            elif acc_type == 'ROBIN':
                mrr_area_div = (3.14*(1.5**2)*1e-6)
                cnn_vdp_unit_area = (mrr_area*N+mrr_area_div*N)*M
                dac_area = M * (2*N) * dac
                total_cnn_units_area = X * \
                    (cnn_vdp_unit_area + pd_area +
                     splitter_area + adc_area + dac_area)
                print('CNN Area', cnn_vdp_unit_area)
                print( S_A_area + eDram_area + sigmoid + router + bus + max_pool_area+io_interface)
                total_area = total_cnn_units_area + S_A_area + \
                eDram_area + sigmoid + router + bus + max_pool_area+io_interface
                print('Total Area', total_area)
            elif acc_type == 'ONNA':
                print('acc_type', acc_type)
                cnn_vdp_unit_area = (mrr_area*N)*M
                dac_area = M * (N) * dac
                total_cnn_units_area = X * \
                    (cnn_vdp_unit_area + pd_area +
                     splitter_area + adc_area + dac_area)
                print('CNN Area', cnn_vdp_unit_area)
                print( S_A_area + eDram_area + sigmoid + router + bus + max_pool_area+io_interface)
                total_area = total_cnn_units_area + S_A_area + \
                eDram_area + sigmoid + router + bus + max_pool_area+io_interface
                print('Total Area', total_area)
            elif acc_type == 'LIGHTBULB':
                mrr_area = 8.03*1e-6
                cnn_vdp_unit_area = (mrr_area*3*N)*M
                dac_area = M * (3*N) * dac
                total_cnn_units_area = X * \
                    (cnn_vdp_unit_area + pd_area +
                     splitter_area + adc_area + dac_area)
                S_A_area = 0.0364 + 0.16 # * Pshifter +Padder
                eDram_area = 0.0143 # * pRacetrack Memory
                print('CNN Area', cnn_vdp_unit_area)
                print( eDram_area + sigmoid + router + bus + max_pool_area+io_interface)
                total_area = total_cnn_units_area +  + \
                eDram_area + sigmoid + router + bus + max_pool_area+io_interface
                print('Total Area', total_area)
            else:
                dac_area = M * (N) * dac
                dac_area_fc = M_FC * N_FC * dac
                cnn_vdp_unit_area = mrr_area*(2*N)*M
                total_cnn_units_area = X * \
                    (cnn_vdp_unit_area + pd_area +
                     splitter_area + adc_area + dac_area)
                total_area = total_cnn_units_area + S_A_area + \
                eDram_area + sigmoid + router + bus + max_pool_area+io_interface
            # print('N', N)
            # print('M', M)
            # print('total_cnn_units_area', total_cnn_units_area)
            # print('S_A_area', S_A_area)
            # print('eDram_area', eDram_area)
            # print('sigmoid', sigmoid)
            # print('router', router)
            # print('bus', bus)
            # print('max_pool_area', max_pool_area)
            # print('io_interface', io_interface)
            

            return total_area

        elif TYPE == 'MMA':
            premux = 0  # um
            WDM = 0
            preweight = 100
            weightblk = 2 * (radius + pitch) + 30
            width = premux + WDM + preweight + weightblk
            height = M * (N*(radius + pitch)+pitch)

            cnn_vdp_unit_area = height * width * 1e-6  # mm2
            fc_weightblk = 2 * (radius + pitch) + 30
            fc_width = premux + WDM + preweight + fc_weightblk
            fc_height = M_FC * (N_FC*(radius + pitch)+pitch)
            fc_vdp_unit_area = fc_width * fc_height * 1e-6  # mm2
            splitter_area = M * splitter
            splitter_area_FC = 0
            pd_area = M * pd
            pd_area_fc = M_FC * pd
            adc_area = M * N * adc
            adc_area_fc = M_FC * N_FC * adc
            dac_area = M * N * dac * 2
            dac_area_fc = M_FC * N_FC * dac

            total_cnn_units_area = X * \
                (cnn_vdp_unit_area + pd_area + splitter_area + adc_area + dac_area)

            total_fc_units_area = Y * \
                (fc_vdp_unit_area + pd_area_fc +
                 splitter_area_FC + adc_area_fc + dac_area_fc)

            total_area = total_cnn_units_area + total_fc_units_area + S_A_area + \
                eDram_area + sigmoid + router + bus + max_pool_area+io_interface

            return total_area
