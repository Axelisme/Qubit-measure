from copy import deepcopy
from attrdict import AttrDict


class InstrManager:
    # template
    CONFIG_TEMP = {"server_ip": None, "instruments": {}}

    def __init__(self, server_ip: str = None, timeout=100):
        """
        Create an InstrManager object that can be used to remote instruments
        via InstrumentServer.

        Example
        ----------
        # Create an InstrManager object that can remote instruments
        a = InstrManager(server_ip='192.168.10.38')

        # Remotely add instruments to the "InstrumentServer"
        a.add_instrument(sHardware='Rohde&Schwarz RF Source',
                          dComCfg={'name': 'sgs114214',
                                  'address': '192.168.10.47',
                                  'interface': 'TCPIP'},
                          sDescription='sgs114214 (Cavity2) for cavity2 upconversion')

        a.add_instrument(sHardware='Rohde&Schwarz RF Source',
                          dComCfg={'name': 'sgs114212',
                                  'address': '192.168.10.43',
                                  'interface': 'TCPIP'},
                          sDescription='sgs114212 (LO) for ext up/down-conversion.')
        a.add_instrument(sHardware='Rohde&Schwarz RF Source',
                          dComCfg={'name': 'sgs114213',
                                  'address': '192.168.10.44',
                                  'interface': 'TCPIP'},
                          sDescription='sgs114213 (Cavity1) for cavity1 upconversion.')

        # Set/Get the action of the instrument
        for dev_name in a.ctrl.keys():
            instrObj = a.ctrl[dev_name]
            instrObj.setValue('Power', -40)
            instrObj.setValue('Frequency', 1e9)
            print(f'{dev_name}:  Power ->    {instrObj.getValue("Power")}')
            print(f'{dev_name}:  Frequency-> { instrObj.getValue("Frequency")}')

        # View the current config of instance object "a"
        pprint(a.config)

        # Export config in json format
        a.export_config('fast_setup/rs_rfsource_fast_setup.json')


        print('')


        # InstrManager will load the configuration to set up instruments
        b = InstrManager()
        b.load_config('fast_setup/rs_rfsource_fast_setup.json')

        # View the current config of instance object "b"
        pprint(b.config)

        # Set/Get the action of the instrument
        print('Frequency: ', b.ctrl.sgs114214.getValue('Frequency'))
        b.ctrl.sgs114214.setValue('Frequency', 3.14e9)

        # View the current config of instance object "b"
        print(b.config)

        # Export config in json format
        b.export_config('fast_setup/test_modify.json')

        Parameters
        ----------
        server_ip : str, optional
            ip address of Labber InstrumentServer.
            The default is None.



        Returns
        -------
        None.

        """
        self._client = None
        self._server_ip = server_ip
        self._timeout = timeout
        self._cfg = None
        self._instr_on_server_dict = {}
        self.ctrl = AttrDict({})

        if isinstance(server_ip, str):
            self.initialize()

    @property
    def instr_handler(self) -> dict:
        """
        Handler that can control instrument behavior.

        Returns
        -------
        dict
            {instr_name: instrObj}.

            After creating ctrl object, user can control the instrument
            by follow these commands.

            Example:
            -------
            >> a.instr_handler # get the handler

            {'sgs114214': <Labber._include38.LabberClient.InstrumentClient at 0x13ada58b700>,
             'sgs114212': <Labber._include38.LabberClient.InstrumentClient at 0x13ada597760>,
             'sgs114213': <Labber._include38.LabberClient.InstrumentClient at 0x13ada59fe50>}

            >> handler = a.instr_hander
            >> handler['sgs114214'].getValue('Frequency') # get Frequency
            3140000000.0

        """

        return dict(self.ctrl)

    @property
    def config(self) -> dict:
        """
        config

        Returns
        -------
        dict
            config of current setup.

        """
        self._update_instr_status2cfg()
        return dict(self._cfg)

    def initialize(self) -> None:
        """
        Create a client object from InstrumentServer, get the list of
        instruments from InstrumentServer and create the cfg according to the
        CONFIG_TEMP.

        Returns
        -------
        None.

        """
        self._get_response_from_server()

        self._chk_instrs_exist_on_server()

        self._create_cfg_template()

    def _create_cfg_template(self):
        if self._cfg is not None:
            # print('cfg exists in this object!')
            return
        # Copy the template and update server_ip
        self._cfg = deepcopy(self.__class__.CONFIG_TEMP)
        self._cfg["server_ip"] = self._server_ip

    def _get_response_from_server(self):
        if self._server_ip is None:
            self._server_ip = self._cfg.server_ip
        # Connect to Labber Instrument server
        # and return a Labber client object.
        from labber_api import Labber

        self._client = Labber.connectToServer(
            address=self._server_ip, timeout=self._timeout
        )

    def _chk_instrs_exist_on_server(self):
        # Get a list of instruments present on the Labber instrument server
        for instr_table in self._client.getListOfInstruments():
            sHardware = instr_table[0]
            instr_name = instr_table[1]["name"]
            address = instr_table[1]["address"]
            if instr_name == "":
                continue
            # Update a dict of instruments present
            self._instr_on_server_dict[instr_name] = {
                "sHardware": sHardware,
                "address": address,
            }

    def get_instr_address_list(self) -> list:
        """
        Get a list of instrument addresses.

        Returns
        -------
        instr_addrlist : list
            a list of addresses.

        """
        self._chk_instrs_exist_on_server()
        instr_addrlist = []
        for instr in self._instr_on_server_dict.values():
            instr_addrlist.append(instr["address"])
        return instr_addrlist

    def add_instrument(
        self,
        sHardware: str,
        dComCfg: dict,
        sDescription: str = None,
        silent: bool = False,
    ):
        """
        Add a specific instrument and activate it.

        Example
        -------
        a.add_instrument(sHardware='Rohde&Schwarz RF Source',
                          dComCfg={'name': 'sgs114214',
                                  'address': '192.168.10.47',
                                  'interface': 'TCPIP'},
                          sDescription='sgs114214 (Cavity2) for cavity2 upconversion')

        Parameters
        ----------
        sHardware : str
            hardware name, for example: 'Rohde&Schwarz RF Source'

        dComCfg : dict
            communication dict, for example: {'name': 'sgs114214',
                                              'address': '192.168.10.47',
                                              'interface': 'TCPIP'},

        sDescription : str, optional
            description, for example: 'sgs114214 (Cavity2) for cavity2 upconversion' .
            The default is None.

        Raises
        ------
        KeyError
            check if the keys of dComCfg exist in
            ['name', 'address', 'interface'].

        Returns
        -------
        None.

        """
        # Check key-value pairs
        comCfgChkList = ("name", "address", "interface")
        for sItemCheck in comCfgChkList:
            if sItemCheck not in dComCfg.keys():
                raise KeyError(f"dConfig requires key-value pairs\nkey:{sItemCheck}")

        # Scan the current instrument list to check if there is
        # an assigned instrument on the server
        instr_address_list = self.get_instr_address_list()
        if dComCfg["address"] not in instr_address_list:
            self._client.createInstrument(sHardware, dComCfg)
        # Activate instrument
        self._start_instrument(sHardware, dComCfg, sDescription, silent=silent)

    def _start_instrument(
        self,
        sHardware: str,
        dComCfg: dict,
        sDescription: str = None,
        silent: bool = False,
    ):
        # Connect to an instrument object on the instrument server
        instr_name = dComCfg["name"]
        instrObj = self._client.connectToInstrument(sHardware, dComCfg)
        instrObj.startInstrument()
        if not silent:
            print(f"Activate {instr_name} successfully.")

        # Get config from InstrumentServer and update it to setup_config
        self._add_instr2config(
            sHardware,
            dComCfg,
            setup_config=instrObj.getInstrConfig(),
            sDescription=sDescription,
        )

        # Obtain control authority of instrument
        self.ctrl[instr_name] = instrObj

    def _add_instr2config(
        self,
        sHardware: str,
        dComCfg: str,
        sDescription: str = None,
        setup_config: dict = None,
    ):
        instr_name = dComCfg["name"]
        if sDescription is None:
            sDescription = f"You can write some purposes of {instr_name}."

        self._cfg["instruments"][instr_name] = {
            "sHardware": sHardware,
            "description": sDescription,
            "communication": {
                "address": dComCfg["address"],
                "interface": dComCfg["interface"],
            },
            "quantity": setup_config,
        }

    def _update_instr_status2cfg(self):
        for instr_name, ctrlObj in self.ctrl.items():
            curQuantity = ctrlObj.getInstrConfig()
            self._cfg["instruments"][instr_name]["quantity"] = curQuantity
