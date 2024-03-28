import framework
import tasks


from config.default import register_args


def main():
    helper = framework.helpers.TrainingHelper(register_args=register_args)
    task = tasks.RLTaskCRF(helper)
    task.train()
    task.test()
    helper.finish()

if __name__ == '__main__':
    main()
