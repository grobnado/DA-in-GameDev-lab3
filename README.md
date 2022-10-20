# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Бурдюгова Екатерина Дмитриевна
- ХС21
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
познакомиться с программными средствами для организции передачи данных между инструментами google, Python и Unity

## Задание 1
### Реализовать систему машинного обучения в связке Python — Google-таблицы — Unity.
Ход работы:
- В проекте unity добавила ml-agents-release_19/com.unity.ml-agents/package.json и ml-agents-release_19/com.unity.ml-agents.extensions/package.json
![image](https://user-images.githubusercontent.com/114610399/196995698-54a95a0c-009a-4292-8f3b-ed10182a7424.png)
- Создала окружение и скачала mlagents и torch
- Создала пространство и объекты на нем
![image](https://user-images.githubusercontent.com/114610399/196996746-5b179205-58e7-4c4b-b7c4-1fac12f8a0a0.png)
- Добавим скрипт сферы RollerAgent.cs
```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```
- Добавила скрипт RollerAgent.cs к сфере и также компоненты Decision Requester, Behavior Parameters
![image](https://user-images.githubusercontent.com/114610399/196997681-b5537efb-e7ed-4647-a6bd-9dcc207a1e18.png)
- В корень проекта добавила файл конфигурации нейронной сети и запустила работу ml-agena
![image](https://user-images.githubusercontent.com/114610399/196998250-7ee0bb6f-775b-4cca-833c-28e6faacbfab.png)
- Создала несколько копий моделей TargetArea и выполнила обучение
![image](https://user-images.githubusercontent.com/114610399/196999504-b8bdd50d-0662-4012-bfe2-f81fcc452ddb.png)
### При большем количестве копий модель обучается быстрее за меньшее время итераций

## Задание 2
### Подробно опишите каждую структуру сборки нейронной сети. Самостоятельно найти информацию о компонентах запрашивающего решения, параметрах поведения, дополнительных сферах.
Ход работы:
```py
behaviors:
  RollerBall: # id агента
    trainer_type: ppo # режим обучения (Proximal Policy Optimization)
    hyperparameters:
      batch_size: 10 # количество опыта на каждой итерации
      buffer_size: 100 # количество опыта, которое нужно набрать перед обновлением модели
      learning_rate: 3.0e-4 # начальная скорость обучения
      beta: 5.0e-4 # сила регуляции энтропии, увеличивает случайность действий
      epsilon: 0.2 # порог расхождений между старой и новой политиками при обновлении
      lambd: 0.99 # насколько агент полагается на свою текущую оценку значений при расчете предсказаний
      num_epoch: 3 # количество проходов, которые необходимо выполнить через буфер опыта при выполнении оптимизации
      learning_rate_schedule: linear # определяет как скорость обучения изменяется с течением времени
                                     # linear линейно уменьшает скорость
    network_settings:
      normalize: false # отключаем нормализацию входных данных
      hidden_units: 128 # количество нейронов в скрытых слоях сети
      num_layers: 2 # количество скрытых слоёв в сети
    reward_signals:
      extrinsic:
        gamma: 0.99 # коэффициент скидки для будущих вознаграждений
        strength: 1.0 # коэффициент на который умножается вознаграждение
    max_steps: 500000 # общее количество шагов, которые должны быть выполнены в среде до завершения обучения
    time_horizon: 64 # сколько опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер
    summary_freq: 10000 # количество опыта, который необходимо собрать перед созданием и отображением статистики обучения
```
Decision Requester - запрос на принятие решения вызывает CollectObservation, а затем получает последнее действие в OnActionReceived, основанное на этом новом собранном наблюдении. С действиями из TakeActionBetweenDecisions он только снова вызовет OnActionReceived без сбора новых наблюдений и выведет последнее действие, которое он получил от NN.
Behavior Parameters - определяет, как Агент принимает решения в зависимости от того, какой тип поведения будет использоваться: уже обученная модель или удалённый процесс обучения.
## Задание 3
### Доработать сцену и обучить ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и впервом задании, случайно изменять кооринаты на плоскости.
Ход работы:
- Добавила еще один куб и изменила код для двух таргетов
```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class Move : Agent
{
    [SerializeField] private GameObject goldMine;
    [SerializeField] private GameObject village;
    private float speedMove;
    private float timeMining;
    private float month;
    private bool checkMiningStart = false;
    private bool checkMiningFinish = false;
    private bool checkStartMonth = false;
    private bool setSensor = true;
    private float amountGold;
    private float pickaxeСost;
    private float profitPercentage;
    private float[] pricesMonth = new float[2];
    private float priceMonth;
    private float tempInf;

    // Start is called before the first frame update
    public override void OnEpisodeBegin()
    {
        // If the Agent fell, zero its momentum
        if (this.transform.localPosition != village.transform.localPosition)
        {
            this.transform.localPosition = village.transform.localPosition;
        }
        checkMiningStart = false;
        checkMiningFinish = false;
        checkStartMonth = false;
        setSensor = true;
        priceMonth = 0.0f;
        pricesMonth[0] = 0.0f;
        pricesMonth[1] = 0.0f;
        tempInf = 0.0f;
        month = 1;
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(speedMove);
        sensor.AddObservation(timeMining);
        sensor.AddObservation(amountGold);
        sensor.AddObservation(pickaxeСost);
        sensor.AddObservation(profitPercentage);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (month < 3 || setSensor == true)
        {
            speedMove = Mathf.Clamp(actionBuffers.ContinuousActions[0], 1f, 10f);
            Debug.Log("SpeedMove: " + speedMove);
            timeMining = Mathf.Clamp(actionBuffers.ContinuousActions[1], 1f, 10f);
            Debug.Log("timeMining: " + timeMining);
            setSensor = false;
            if (checkStartMonth == false)
            {
                Debug.Log("Start Coroutine StartMonth");
                StartCoroutine(StartMonth());
            }

            if (transform.position != goldMine.transform.position & checkMiningFinish == false)
            {
                transform.position = Vector3.MoveTowards(transform.position, goldMine.transform.position, Time.deltaTime * speedMove);
            }

            if (transform.position == goldMine.transform.position & checkMiningStart == false)
            {
                Debug.Log("Start Coroutine StartGoldMine");
                StartCoroutine(StartGoldMine());
            }

            if (transform.position != village.transform.position & checkMiningFinish == true)
            {
                transform.position = Vector3.MoveTowards(transform.position, village.transform.position, Time.deltaTime * speedMove);
            }

            if (transform.position == village.transform.position & checkMiningStart == true)
            {
                checkMiningFinish = false;
                checkMiningStart = false;
                setSensor = true;
                amountGold = Mathf.Clamp(actionBuffers.ContinuousActions[2], 1f, 10f);
                Debug.Log("amountGold: " + amountGold);
                pickaxeСost = Mathf.Clamp(actionBuffers.ContinuousActions[3], 100f, 1000f);
                Debug.Log("pickaxeСost: " + pickaxeСost);
                profitPercentage = Mathf.Clamp(actionBuffers.ContinuousActions[4], 0.1f, 0.5f);
                Debug.Log("profitPercentage: " + profitPercentage);

                if (month != 2)
                {
                    priceMonth = pricesMonth[0] + ((pickaxeСost + pickaxeСost * profitPercentage) / amountGold);
                    pricesMonth[0] = priceMonth;
                    Debug.Log("priceMonth: " + priceMonth);
                }
                if (month == 2)
                {
                    priceMonth = pricesMonth[1] + ((pickaxeСost + pickaxeСost * profitPercentage) / amountGold);
                    pricesMonth[1] = priceMonth;
                    Debug.Log("priceMonth: " + priceMonth);
                }

            }
        }
        else
        {
            tempInf = ((pricesMonth[1] - pricesMonth[0]) / pricesMonth[0]) * 100;
            if (tempInf <= 6f)
            {
                SetReward(1.0f);
                Debug.Log("True");
                Debug.Log("tempInf: " + tempInf);
                EndEpisode();
            }
            else
            {
                SetReward(-1.0f);
                Debug.Log("False");
                Debug.Log("tempInf: " + tempInf);
                EndEpisode();
            }
        }
    }

    IEnumerator StartGoldMine()
    {
        checkMiningStart = true;
        yield return new WaitForSeconds(timeMining);
        Debug.Log("Mining Finish");
        checkMiningFinish = true;
    }

    IEnumerator StartMonth()
    {
        checkStartMonth = true;
        yield return new WaitForSeconds(60);
        checkStartMonth = false;
        month++;

    }
}
```
- Снова создаем несколько моделей и получаем результат

### Вывод
Игровой баланс - воспринимаемый игроком уровень равновесия между параметрами игры с которыми взаимодействует игрок. Хороший баланс должен поддерживать уровень веселья в небольшом диапазоне, чтобы было и сложно и просто одновременно. При этом игрок должен ощущать, что игра честная по отношению к нему.
Существует 2 основных игровых системы: симметричные игры, где обе стороны начинают с одинаковыми условиями, и асимметричные.
Силу балансируют с помощью транзитивного, интранзитивного и некомпаративного методов:
Транзитивный - прямое сравнение или соотношение характеристик объектов.
Интразитивный - подразумевает сбор статистики об успешности столкновений между разными персонажами в поисках самых слабых и сильных.
Некомпаративный - подразумевает, что ничего с балансом осознанно сделать нельзя из-за принципиальной несравнимости абилок.
Машинное обучение способно снизить проблему долгой игры для наведения баланса, так как это будет происходить банально быстрее.
Я научилась работать с ML агентом, поняла как пишется код для обучения ИИ, поняла как получить результат обучения. Узнала, что чем больше моделей задействованы в обучении, тем лучше будет результат обучения. Разобралась как создать пространство под агента, и как пользоваться консолью в anaconda.
