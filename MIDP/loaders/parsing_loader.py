from .. import parsers


class ParsingLoader:

    # def __init__(self, *multiple_parser_configs, **single_parser_config):
    def __init__(self, parser_config=None):
        parser_map = dict()

        def set_parser_map(parser_config):
            config = parser_config.copy()
            parser = getattr(parsers, config.pop('name'))(**config)
            parser_map.update(dict(zip(
                parser.data_list,
                [parser] * parser.n_data
            )))

            # check and preserve properties
            for p in [
                'n_labels',
                'ROIs',
            ]:
                try:
                    assert getattr(self, p) == getattr(parser, p), p
                except AttributeError:
                    setattr(self, p, getattr(parser, p))

        if not isinstance(parser_config, list):
            parser_config = [parser_config]

        for _parser_config in parser_config:
            set_parser_map(_parser_config)

        # if single_parser_config:
        #     set_parser_map(single_parser_config)
        # else:
        #     for parser_config in multiple_parser_configs:
        #         set_parser_map(parser_config)

        self._data_list = list(parser_map.keys())

        def mirror_func(func_name):
            return lambda idx: getattr(parser_map[idx], func_name)(idx)

        for func_name in [
            'get_image',
            'get_label',
            'get_image_shape',
            'get_label_shape'
        ]:
            setattr(self, func_name, mirror_func(func_name))

    @property
    def n_data(self):
        return len(self._data_list)

    @property
    def data_list(self):
        return self._data_list

    def set_data_list(self, new_list):
        assert set(new_list).issubset(self._data_list), new_list
        self._data_list = new_list
